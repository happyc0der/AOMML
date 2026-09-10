"""Does it actually solve the problem? These are the load-bearing tests.

Correctness here is anchored to two external references: scikit-learn's
coordinate descent (an independent solver for the same objective) and the known
ground-truth ``w_true`` baked into the synthetic data.
"""

import pytest
import torch

from aomml.data import make_sparse_regression
from aomml.models import LinearModel, ReLUMLP
from aomml.objectives import lasso_objective
from aomml.optimizers import SSGD, SSGDMomentum
from aomml.train import train
from conftest import sklearn_reference


def test_lasso_matches_sklearn(synth, lam, lasso_loss):
    """SSGD must reach the same optimum an independent solver finds.

    This is the single strongest piece of evidence that the optimizer, the
    objective and the training loop are all correct together.
    """
    w_sk, _, f_star = sklearn_reference(synth, lam)

    model = LinearModel(synth.X.shape[1])
    history = train(
        model, lasso_loss, synth.X, synth.y,
        SSGD(model.parameters(), lr=0.3, schedule="inv_sqrt"),
        epochs=300, batch_size=32, eval_every=50, seed=0,
    )

    assert history.final_best - f_star < 1e-3
    assert (model.weight.detach() - w_sk).abs().max() < 1e-2


def test_momentum_matches_sklearn(synth, lam, lasso_loss):
    w_sk, _, f_star = sklearn_reference(synth, lam)

    model = LinearModel(synth.X.shape[1])
    history = train(
        model, lasso_loss, synth.X, synth.y,
        SSGDMomentum(model.parameters(), lr=0.05, beta=0.9, schedule="inv_sqrt"),
        epochs=300, batch_size=32, eval_every=50, seed=0,
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
        model, lasso_loss, synth.X, synth.y,
        SSGD(model.parameters(), lr=0.3, schedule="inv_sqrt"),
        epochs=300, batch_size=32, eval_every=100, seed=0,
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
            model, loss, data.X, data.y,
            SSGD(model.parameters(), lr=2.0, schedule=schedule),
            epochs=epochs, batch_size=8, eval_every=50, seed=0,
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


def test_momentum_accelerates_on_ill_conditioned_problem():
    """Heavy-ball's advantage appears when the problem is badly conditioned."""
    data = make_sparse_regression(
        n=300, d=30, k=30, noise_std=0.0, condition_number=200.0, seed=11
    )
    loss = lambda m, Xb, yb: lasso_objective(m.weight, Xb, yb, 0.0, m.bias)  # noqa: E731

    def final(beta):
        model = LinearModel(30)
        opt = (
            SSGD(model.parameters(), lr=1.0, schedule="constant")
            if beta == 0
            else SSGDMomentum(model.parameters(), lr=1.0, beta=beta, schedule="constant")
        )
        h = train(
            model, loss, data.X, data.y, opt,
            epochs=3000, batch_size=300, eval_every=50, seed=0,
        )
        return h.final_best

    plain, mom, high_mom = final(0.0), final(0.9), final(0.99)

    assert mom < plain / 10
    assert high_mom < mom


def test_objective_is_not_monotone(synth, lam, lasso_loss):
    """The subgradient method is not a descent method -- f(w_k) genuinely rises.

    If this ever passes trivially (perfectly monotone), the step size has become
    so small the method is no longer stochastic subgradient descent.
    """
    model = LinearModel(synth.X.shape[1])
    h = train(
        model, lasso_loss, synth.X, synth.y,
        SSGD(model.parameters(), lr=0.3, schedule="inv_sqrt"),
        epochs=200, batch_size=32, eval_every=25, seed=0,
    )

    increases = sum(1 for a, b in zip(h.f, h.f[1:]) if b > a)
    assert increases > 0
    # ...while the tracked best iterate is monotone by construction.
    assert all(b <= a for a, b in zip(h.f_best, h.f_best[1:]))


def test_mlp_overfits_small_batch():
    """A network that cannot memorise 32 examples is mis-wired."""
    torch.manual_seed(0)
    X = torch.randn(32, 20)
    y = torch.randint(0, 5, (32,))
    model = ReLUMLP((20, 64, 5), dtype=torch.float32)
    loss = lambda m, Xb, yb: torch.nn.functional.cross_entropy(m(Xb), yb)  # noqa: E731

    h = train(
        model, loss, X, y,
        SSGDMomentum(model.parameters(), lr=0.1, beta=0.9, schedule="constant"),
        epochs=600, batch_size=32, eval_every=50, seed=0,
    )
    assert h.final_best < 0.05
