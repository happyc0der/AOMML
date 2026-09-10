"""Stochastic subgradient descent, with and without momentum.

Both are real ``torch.optim.Optimizer`` subclasses, so they carry proper state
and interoperate with the rest of torch rather than being loop-local update rules.

Step-size schedules matter more here than they do for smooth problems. For a
non-smooth convex objective a *constant* step converges only to a neighbourhood
of the optimum whose radius scales with the step; actual convergence requires
``alpha_k -> 0`` with ``sum(alpha_k) = inf``. The ``inv_sqrt`` schedule
(``alpha_0 / sqrt(k+1)``) gives the standard ``O(1/sqrt(k))`` rate and is the
default for that reason.
"""

from __future__ import annotations

import math

import torch
from torch.optim import Optimizer

__all__ = ["SSGD", "SSGDMomentum", "schedule_factor", "SCHEDULES"]

SCHEDULES = ("constant", "inv_sqrt", "inv")


def schedule_factor(schedule: str, k: int) -> float:
    """Multiplier applied to the base step size at iteration ``k`` (0-indexed)."""
    if schedule == "constant":
        return 1.0
    if schedule == "inv_sqrt":
        return 1.0 / math.sqrt(k + 1)
    if schedule == "inv":
        return 1.0 / (k + 1)
    raise ValueError(f"unknown schedule {schedule!r}, expected one of {SCHEDULES}")


class _ScheduledOptimizer(Optimizer):
    """Shared step-size bookkeeping for the two methods below."""

    def _validate(self, lr: float, schedule: str) -> None:
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if schedule not in SCHEDULES:
            raise ValueError(f"unknown schedule {schedule!r}, expected one of {SCHEDULES}")

    def current_lr(self, group_index: int = 0) -> float:
        """The step size that the *next* call to ``step`` will apply."""
        group = self.param_groups[group_index]
        return group["lr"] * schedule_factor(group["schedule"], group["k"])


class SSGD(_ScheduledOptimizer):
    """Stochastic subgradient descent: ``w <- w - alpha_k * g_k``.

    ``g_k`` is any subgradient of the objective at ``w_k`` -- for a differentiable
    objective that is the gradient, and at a non-smooth point (``|w_i| = 0`` in
    LASSO, or a ReLU kink) it is whatever element of the subdifferential autograd
    selects.

    This is deliberately *not* a descent method: ``f(w_k)`` can and does increase
    between iterations. Convergence guarantees apply to the running best iterate
    ``min_{j<=k} f(w_j)`` or the averaged iterate, which is why
    :class:`aomml.train.History` tracks those separately.
    """

    def __init__(self, params, lr: float = 1e-2, schedule: str = "inv_sqrt") -> None:
        self._validate(lr, schedule)
        super().__init__(params, dict(lr=lr, schedule=schedule, k=0))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            alpha = group["lr"] * schedule_factor(group["schedule"], group["k"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.add_(p.grad, alpha=-alpha)
            group["k"] += 1
        return loss


class SSGDMomentum(_ScheduledOptimizer):
    """Stochastic subgradient descent with heavy-ball momentum.

    Two formulations are provided because they are *not* equivalent here:

    ``variant="buffer"`` (default), the form torch itself uses::

        v_{k+1} = beta * v_k + g_k
        w_{k+1} = w_k - alpha_k * v_{k+1}

    ``variant="difference"``, Polyak's original heavy-ball::

        w_{k+1} = w_k - alpha_k * g_k + beta * (w_k - w_{k-1})

    Under a *constant* step size these coincide exactly: telescoping the buffer
    form gives ``w_k - w_{k-1} = -alpha * v_k``, so ``-alpha*beta*v_k`` is
    precisely ``beta*(w_k - w_{k-1})``. Under a *decaying* step size they come
    apart, because the buffer form rescales the entire accumulated history by the
    current ``alpha_k`` while the difference form leaves the previous displacement
    at the scale it was taken with. Since decaying steps are exactly what
    non-smooth convergence requires, the distinction is live for this project
    rather than academic -- ``test_momentum_variants_agree_under_constant_step``
    pins down where they agree, and E4 in the notebook shows where they diverge.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-2,
        beta: float = 0.9,
        schedule: str = "inv_sqrt",
        nesterov: bool = False,
        variant: str = "buffer",
    ) -> None:
        self._validate(lr, schedule)
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"beta must be in [0, 1), got {beta}")
        if variant not in {"buffer", "difference"}:
            raise ValueError(f"unknown variant {variant!r}")
        if nesterov and variant != "buffer":
            raise ValueError("nesterov is only defined for the buffer variant")
        super().__init__(
            params,
            dict(lr=lr, beta=beta, schedule=schedule, k=0,
                 nesterov=nesterov, variant=variant),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            alpha = group["lr"] * schedule_factor(group["schedule"], group["k"])
            beta, variant, nesterov = group["beta"], group["variant"], group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]

                if variant == "buffer":
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    buf = state["momentum_buffer"]
                    buf.mul_(beta).add_(g)
                    # Nesterov looks one momentum step ahead of the current buffer.
                    direction = g.add(buf, alpha=beta) if nesterov else buf
                    p.add_(direction, alpha=-alpha)
                else:  # difference form
                    if "prev" not in state:
                        state["prev"] = p.detach().clone()
                    displacement = p.detach() - state["prev"]
                    state["prev"] = p.detach().clone()
                    p.add_(g, alpha=-alpha).add_(displacement, alpha=beta)

            group["k"] += 1
        return loss
