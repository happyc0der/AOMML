"""The training loop's bookkeeping: History, averaging, and evaluation."""

import pytest
import torch

from aomml.data import make_sparse_regression
from aomml.models import LinearModel, ReLUMLP
from aomml.objectives import lasso_objective
from aomml.optimizers import SSGD
from aomml.train import History, evaluate_accuracy, train


def _run(**kw):
    data = make_sparse_regression(n=120, d=20, k=3, seed=9)
    loss = lambda m, Xb, yb: lasso_objective(m.weight, Xb, yb, 0.02, m.bias)  # noqa: E731
    model = LinearModel(20)
    return train(
        model,
        loss,
        data.X,
        data.y,
        SSGD(model.parameters(), lr=0.1, schedule="inv_sqrt"),
        epochs=kw.pop("epochs", 20),
        batch_size=16,
        eval_every=5,
        seed=3,
        **kw,
    )


def test_history_best_is_monotone_non_increasing():
    h = _run()
    assert all(b <= a for a, b in zip(h.f_best, h.f_best[1:], strict=False))
    assert h.final_best == min(h.f)


def test_history_records_all_series_at_equal_length():
    h = _run()
    n = len(h)
    assert n > 1
    assert (
        len(
            {
                len(h.step),
                len(h.f),
                len(h.f_best),
                len(h.f_avg),
                len(h.grad_norm),
                len(h.lr),
                len(h.elapsed),
            }
        )
        == 1
    )


def test_steps_to_reach_finds_first_crossing():
    h = History(step=[0, 10, 20, 30], f=[9.0, 5.0, 2.0, 1.0], f_best=[9.0, 5.0, 2.0, 1.0])
    assert h.steps_to_reach(5.0) == 10
    assert h.steps_to_reach(2.5) == 20
    assert h.steps_to_reach(1.0) == 30


def test_steps_to_reach_returns_none_when_never_reached():
    """A run that never hits the target must report None, not the last step --
    this is what makes the constant-step 'never converges' result readable."""
    h = History(step=[0, 10], f=[9.0, 5.0], f_best=[9.0, 5.0])
    assert h.steps_to_reach(0.001) is None


def test_steps_to_reach_can_use_raw_iterate():
    h = History(step=[0, 10, 20], f=[9.0, 1.0, 7.0], f_best=[9.0, 1.0, 1.0])
    assert h.steps_to_reach(2.0, use_best=False) == 10


def test_polyak_average_differs_from_raw_iterate():
    """The averaged iterate is tracked separately, not aliased to f(w_k)."""
    h = _run(epochs=40)
    assert h.f_avg != h.f


def test_training_restores_parameters_after_averaged_evaluation():
    """Evaluating at the averaged iterate must not disturb the live model."""
    data = make_sparse_regression(n=80, d=10, k=2, seed=4)
    loss = lambda m, Xb, yb: lasso_objective(m.weight, Xb, yb, 0.02, m.bias)  # noqa: E731
    model = LinearModel(10)
    train(
        model,
        loss,
        data.X,
        data.y,
        SSGD(model.parameters(), lr=0.1),
        epochs=10,
        batch_size=16,
        eval_every=1,
        seed=0,
    )
    after = model.weight.detach().clone()
    # Re-evaluating must be a pure read.
    with torch.no_grad():
        loss(model, data.X, data.y)
    assert torch.equal(model.weight.detach(), after)


def test_evaluate_accuracy_on_known_predictions():
    """A model that is right on exactly 3 of 4 examples must score 0.75."""

    class Fixed(torch.nn.Module):
        def forward(self, X):
            return X

    logits = torch.tensor([[9.0, 0.0], [9.0, 0.0], [0.0, 9.0], [0.0, 9.0]])
    labels = torch.tensor([0, 0, 1, 0])
    assert evaluate_accuracy(Fixed(), logits, labels) == pytest.approx(0.75)


def test_evaluate_accuracy_batching_matches_single_pass():
    torch.manual_seed(0)
    net = ReLUMLP((6, 8, 3))
    X, y = torch.randn(50, 6), torch.randint(0, 3, (50,))
    assert evaluate_accuracy(net, X, y, batch_size=7) == pytest.approx(
        evaluate_accuracy(net, X, y, batch_size=1000)
    )


def test_evaluate_accuracy_leaves_model_in_training_mode():
    net = ReLUMLP((6, 8, 3))
    net.train()
    evaluate_accuracy(net, torch.randn(10, 6), torch.randint(0, 3, (10,)))
    assert net.training


def test_train_rejects_mismatched_lengths():
    model = LinearModel(5)
    loss = lambda m, Xb, yb: lasso_objective(m.weight, Xb, yb, 0.0, m.bias)  # noqa: E731
    with pytest.raises(ValueError):
        train(
            model,
            loss,
            torch.randn(10, 5, dtype=torch.float64),
            torch.randn(9, dtype=torch.float64),
            SSGD(model.parameters()),
        )


def test_train_rejects_bad_batch_size():
    model = LinearModel(5)
    loss = lambda m, Xb, yb: lasso_objective(m.weight, Xb, yb, 0.0, m.bias)  # noqa: E731
    with pytest.raises(ValueError):
        train(
            model,
            loss,
            torch.randn(10, 5, dtype=torch.float64),
            torch.randn(10, dtype=torch.float64),
            SSGD(model.parameters()),
            batch_size=0,
        )
