"""Does it actually solve the problem? These are the load-bearing tests.

Correctness here is anchored to two external references: scikit-learn's
coordinate descent (an independent solver for the same objective) and the known
ground-truth ``w_true`` baked into the synthetic data.
"""

import pytest
import torch
from conftest import sklearn_reference

from aomml.data import make_sparse_regression
from aomml.models import LinearModel, ReLUMLP
from aomml.objectives import lasso_objective
from aomml.optimizers import SSGD, SSGDMomentum
from aomml.train import train


def test_lasso_matches_sklearn(synth, lam, lasso_loss):
    """SSGD must reach the same optimum an independent solver finds.

    This is the single strongest piece of evidence that the optimizer, the
    objective and the training loop are all correct together.
    """
    w_sk, _, f_star = sklearn_reference(synth, lam)

    model = LinearModel(synth.X.shape[1])
    history = train(
        model,
        lasso_loss,
        synth.X,
        synth.y,
        SSGD(model.parameters(), lr=0.3, schedule="inv_sqrt"),
        epochs=300,
        batch_size=32,
        eval_every=50,
        seed=0,
    )

    assert history.final_best - f_star < 1e-3
    assert (model.weight.detach() - w_sk).abs().max() < 1e-2


def test_momentum_matches_sklearn(synth, lam, lasso_loss):
    w_sk, _, f_star = sklearn_reference(synth, lam)

    model = LinearModel(synth.X.shape[1])
    history = train(
        model,
        lasso_loss,
        synth.X,
        synth.y,
        SSGDMomentum(model.parameters(), lr=0.05, beta=0.9, schedule="inv_sqrt"),
        epochs=300,
        batch_size=32,
        eval_every=50,
        seed=0,
    )

    assert history.final_best - f_star < 1e-3
    assert (model.weight.detach() - w_sk).abs().max() < 1e-2


def test_support_recovery(synth, lam, lasso_loss):
    """Every truly non-zero coefficient must be recovered.

    Note the asymmetry: plain subgradient descent does *not* drive coefficients
    to exact zero the way coordinate descent or a proximal method does, so the
    recovered support is thresholded and may be slightly larger than the truth.
    That is a property of the method, not a bug.
    """
    model = LinearModel(synth.X.shape[1])
    train(
        model,
        lasso_loss,
        synth.X,
        synth.y,
        SSGD(model.parameters(), lr=0.3, schedule="inv_sqrt"),
        epochs=300,
        batch_size=32,
        eval_every=100,
        seed=0,
    )

    recovered = set(torch.nonzero(model.weight.detach().abs() > 1e-2).flatten().tolist())
    truth = set(synth.support.tolist())

    assert truth <= recovered, f"missed true coefficients: {truth - recovered}"
    assert len(recovered) <= len(truth) + 3


def test_diminishing_step_beats_constant_step():
    """A constant step converges only to a *neighbourhood* of the optimum.

    Evidence: quadrupling the iteration budget must not improve the constant-step
    run (it has stalled at the noise floor), while the 1/sqrt(k) run keeps
    descending. This is the central theoretical claim of the project.
    """
    data = make_sparse_regression(n=200, d=40, k=5, noise_std=0.5, seed=3)
    lam = 0.02
    loss = lambda m, Xb, yb: lasso_objective(m.weight, Xb, yb, lam, m.bias)  # noqa: E731
    _, _, f_star = sklearn_reference(data, lam)

    def gap(schedule, epochs):
        model = LinearModel(40)
        h = train(
            model,
            loss,
            data.X,
            data.y,
            SSGD(model.parameters(), lr=2.0, schedule=schedule),
            epochs=epochs,
            batch_size=8,
            eval_every=50,
            seed=0,
        )
        return h.final_best - f_star

    const_short, const_long = gap("constant", 100), gap("constant", 400)
    decay_short, decay_long = gap("inv_sqrt", 100), gap("inv_sqrt", 400)

    # The constant step has stalled: 4x the budget buys nothing.
    assert const_long == pytest.approx(const_short, rel=1e-6)
    # The diminishing step is still making progress.
    assert decay_long < decay_short / 2
    # And ends orders of magnitude closer to the true optimum.
    assert decay_long < const_long / 100


def _ill_conditioned_run(beta, lr, epochs=3000):
    """Full-batch run on a smooth, badly-conditioned least-squares problem."""
    data = make_sparse_regression(n=300, d=30, k=30, noise_std=0.0, condition_number=200.0, seed=11)
    loss = lambda m, Xb, yb: lasso_objective(m.weight, Xb, yb, 0.0, m.bias)  # noqa: E731
    model = LinearModel(30)
    opt = (
        SSGD(model.parameters(), lr=lr, schedule="constant")
        if beta == 0
        else SSGDMomentum(model.parameters(), lr=lr, beta=beta, schedule="constant")
    )
    return train(
        model,
        loss,
        data.X,
        data.y,
        opt,
        epochs=epochs,
        batch_size=300,
        eval_every=50,
        seed=0,
    )


def test_momentum_wins_at_fixed_learning_rate():
    """At a fixed alpha_0, higher beta reaches a far lower objective.

    True, but *not* a controlled comparison -- see the next two tests. The
    effective step is alpha_0/(1-beta), so beta=0.99 is silently taking steps
    100x larger. This test exists to pin the observation, not to explain it.
    """
    plain = _ill_conditioned_run(0.0, 1.0).final_best
    mom = _ill_conditioned_run(0.99, 1.0).final_best
    assert mom < plain / 1e6


def test_momentum_advantage_vanishes_at_matched_effective_step():
    """Control for the step size and the per-iteration advantage disappears.

    With alpha_0 = 1 - beta, every beta takes the same effective step, and every
    beta lands in essentially the same place. Momentum is not a better update at
    equal step length.
    """
    plain = _ill_conditioned_run(0.0, 1.0).final_best
    mom = _ill_conditioned_run(0.99, 0.01).final_best
    assert 0.5 < mom / plain < 2.0, (
        f"expected comparable results at matched step, got {plain:.3e} vs {mom:.3e}"
    )


def test_momentum_is_stable_where_plain_ssgd_diverges():
    """The actual mechanism: momentum tolerates a step that plain SSGD cannot.

    Plain subgradient descent is capped by the curvature of the worst-conditioned
    direction. Above that cap it oscillates and never improves on its starting
    point. Momentum damps those oscillations and keeps converging, which is what
    buys the orders-of-magnitude gap in the uncontrolled comparison above.
    """
    data = make_sparse_regression(n=300, d=30, k=30, noise_std=0.0, condition_number=200.0, seed=11)
    loss = lambda m, Xb, yb: lasso_objective(m.weight, Xb, yb, 0.0, m.bias)  # noqa: E731
    with torch.no_grad():
        f_at_zero = float(loss(LinearModel(30), data.X, data.y))

    aggressive = 2.0
    plain = _ill_conditioned_run(0.0, aggressive, epochs=300).final_best
    mom = _ill_conditioned_run(0.9, aggressive, epochs=300).final_best

    assert plain >= f_at_zero, "premise: plain SSGD should fail to improve at this step"
    assert mom < f_at_zero / 100, "momentum should remain stable and converge"


def test_objective_is_not_monotone(synth, lam, lasso_loss):
    """The subgradient method is not a descent method -- f(w_k) genuinely rises.

    If this ever passes trivially (perfectly monotone), the step size has become
    so small the method is no longer stochastic subgradient descent.
    """
    model = LinearModel(synth.X.shape[1])
    h = train(
        model,
        lasso_loss,
        synth.X,
        synth.y,
        SSGD(model.parameters(), lr=0.3, schedule="inv_sqrt"),
        epochs=200,
        batch_size=32,
        eval_every=25,
        seed=0,
    )

    increases = sum(1 for a, b in zip(h.f, h.f[1:], strict=False) if b > a)
    assert increases > 0
    # ...while the tracked best iterate is monotone by construction.
    assert all(b <= a for a, b in zip(h.f_best, h.f_best[1:], strict=False))


def test_mlp_overfits_small_batch():
    """A network that cannot memorise 32 examples is mis-wired."""
    torch.manual_seed(0)
    X = torch.randn(32, 20)
    y = torch.randint(0, 5, (32,))
    model = ReLUMLP((20, 64, 5), dtype=torch.float32)
    loss = lambda m, Xb, yb: torch.nn.functional.cross_entropy(m(Xb), yb)  # noqa: E731

    h = train(
        model,
        loss,
        X,
        y,
        SSGDMomentum(model.parameters(), lr=0.1, beta=0.9, schedule="constant"),
        epochs=600,
        batch_size=32,
        eval_every=50,
        seed=0,
    )
    assert h.final_best < 0.05
