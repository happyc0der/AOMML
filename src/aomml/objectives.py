"""The LASSO objective and its analytic subgradient.

``lasso_objective`` is what training actually optimises (through autograd).
``analytic_lasso_subgradient`` is a closed-form reference used *only* by the
test suite -- asserting the two agree is what makes "is this implemented
correctly" a question with a real answer.
"""

from __future__ import annotations

import torch

__all__ = ["lasso_objective", "analytic_lasso_subgradient", "is_valid_subgradient"]


def _check_shapes(w: torch.Tensor, X: torch.Tensor, y: torch.Tensor) -> None:
    """Reject the shapes that silently produce wrong answers.

    A ``y`` of shape ``(n, 1)`` against a residual of shape ``(n,)`` broadcasts
    to an ``n x n`` matrix and yields a plausible-looking but meaningless loss.
    That failure is silent, so it is worth an explicit guard.
    """
    if w.ndim != 1:
        raise ValueError(f"w must be 1-D, got shape {tuple(w.shape)}")
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {tuple(X.shape)}")
    if y.ndim != 1:
        raise ValueError(
            f"y must be 1-D, got shape {tuple(y.shape)} -- a trailing singleton "
            "dimension here broadcasts the residual into an n x n matrix"
        )
    if X.shape[1] != w.shape[0]:
        raise ValueError(f"X has {X.shape[1]} columns but w has {w.shape[0]} entries")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X has {X.shape[0]} rows but y has {y.shape[0]} entries")


def lasso_objective(
    w: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    lam: float,
    b: torch.Tensor | None = None,
) -> torch.Tensor:
    """``f(w) = (1/2n)||Xw + b - y||^2 + lam * ||w||_1``.

    The intercept ``b`` is deliberately *not* penalised. The original notebook
    had no intercept at all, which forces the response's mean into the penalised
    coefficients and biases the whole solution toward zero.

    The L1 term is non-smooth at zero; torch's ``abs`` backward returns
    ``sign(0) = 0``, which is a valid element of the subdifferential ``[-1, 1]``.
    ``is_valid_subgradient`` below pins that behaviour down rather than trusting it.
    """
    _check_shapes(w, X, y)
    pred = X @ w if b is None else X @ w + b
    n = X.shape[0]
    residual = pred - y
    return 0.5 / n * residual.dot(residual) + lam * w.abs().sum()


def analytic_lasso_subgradient(
    w: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    lam: float,
    b: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Closed-form subgradient of :func:`lasso_objective`. Test oracle only.

    Uses the ``sign(0) = 0`` selection from the subdifferential, matching torch.
    """
    _check_shapes(w, X, y)
    n = X.shape[0]
    residual = (X @ w if b is None else X @ w + b) - y
    grad_w = X.T @ residual / n + lam * torch.sign(w)
    grad_b = residual.sum() / n if b is not None else None
    return grad_w, grad_b


def is_valid_subgradient(
    g: torch.Tensor,
    w: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    lam: float,
    atol: float = 1e-8,
) -> torch.Tensor:
    """Per-coordinate check that ``g`` lies in the subdifferential at ``w``.

    Where ``w_i != 0`` the objective is differentiable and the gradient is
    determined exactly. Where ``w_i == 0`` any value within ``lam`` of the smooth
    part's gradient is admissible, so we check membership in that interval
    rather than equality.
    """
    n = X.shape[0]
    smooth = X.T @ (X @ w - y) / n
    on_zero = w == 0
    exact = smooth + lam * torch.sign(w)
    return torch.where(
        on_zero,
        (g - smooth).abs() <= lam + atol,
        (g - exact).abs() <= atol,
    )
