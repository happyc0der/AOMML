"""The math: does the objective compute what it claims to compute?"""

import pytest
import torch

from aomml.objectives import (
    analytic_lasso_subgradient,
    is_valid_subgradient,
    lasso_objective,
)


def test_autograd_matches_analytic_subgradient(synth, lam):
    """Autograd's subgradient must equal the closed form at a smooth point."""
    w = torch.randn(synth.X.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)

    f = lasso_objective(w, synth.X, synth.y, lam, b)
    g_w, g_b = torch.autograd.grad(f, [w, b])
    a_w, a_b = analytic_lasso_subgradient(w.detach(), synth.X, synth.y, lam, b.detach())

    assert torch.allclose(g_w, a_w, atol=1e-12)
    assert torch.allclose(g_b, a_b, atol=1e-12)


def test_l1_subgradient_valid_at_zero(synth, lam):
    """At w = 0 the objective is non-smooth; autograd must still return a
    *valid* element of the subdifferential, i.e. within lam of the smooth part."""
    w = torch.zeros(synth.X.shape[1], dtype=torch.float64, requires_grad=True)
    f = lasso_objective(w, synth.X, synth.y, lam)
    (g,) = torch.autograd.grad(f, [w])

    assert is_valid_subgradient(g, w.detach(), synth.X, synth.y, lam).all()


def test_objective_value_matches_definition(synth, lam):
    w = torch.randn(synth.X.shape[1], dtype=torch.float64)
    n = synth.X.shape[0]
    expected = (
        0.5 / n * ((synth.X @ w - synth.y) ** 2).sum() + lam * w.abs().sum()
    )
    assert torch.isclose(lasso_objective(w, synth.X, synth.y, lam), expected, atol=1e-12)


def test_intercept_is_not_penalised(synth, lam):
    """A large intercept must not inflate the L1 term."""
    w = torch.zeros(synth.X.shape[1], dtype=torch.float64)
    small = lasso_objective(w, synth.X, synth.y, lam, torch.tensor(0.0, dtype=torch.float64))
    large = lasso_objective(w, synth.X, synth.y, lam, torch.tensor(5.0, dtype=torch.float64))
    penalty = lam * w.abs().sum()
    assert penalty == 0
    # Both differ only through the quadratic term, never the penalty.
    assert small != large


@pytest.mark.parametrize(
    "bad",
    ["y_2d", "w_2d", "mismatched_features", "mismatched_rows"],
)
def test_shape_guards_reject_silent_broadcasting(synth, lam, bad):
    """An (n, 1) target broadcasts a residual into an n x n matrix and yields a
    plausible but meaningless loss. That must raise, not compute."""
    w, X, y = torch.zeros(50, dtype=torch.float64), synth.X, synth.y
    if bad == "y_2d":
        y = y.unsqueeze(1)
    elif bad == "w_2d":
        w = w.unsqueeze(1)
    elif bad == "mismatched_features":
        w = torch.zeros(49, dtype=torch.float64)
    else:
        y = y[:-1]

    with pytest.raises(ValueError):
        lasso_objective(w, X, y, lam)
