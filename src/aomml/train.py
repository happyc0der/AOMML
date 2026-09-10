"""Training loop and history tracking for the subgradient methods.

The subgradient method is *not* a descent method: ``f(w_k)`` is non-monotone and
routinely increases from one iteration to the next. The convergence theory is
stated for the running best iterate ``min_{j<=k} f(w_j)`` and for the averaged
(Polyak) iterate, so :class:`History` records all three. Plotting the raw
iterate alone and concluding the optimiser is broken is the standard mistake
this separation exists to prevent.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import nn

__all__ = ["History", "train", "evaluate_accuracy"]

LossFn = Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class History:
    """Per-evaluation record of a training run."""

    step: list[int] = field(default_factory=list)
    f: list[float] = field(default_factory=list)
    f_best: list[float] = field(default_factory=list)
    f_avg: list[float] = field(default_factory=list)
    grad_norm: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)
    elapsed: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.step)

    @property
    def final_best(self) -> float:
        return self.f_best[-1]

    def steps_to_reach(self, target: float, use_best: bool = True) -> int | None:
        """First recorded step at which the objective drops to ``target``.

        Returns ``None`` if the run never got there -- which is itself the
        finding in the constant-step-size experiment.
        """
        series = self.f_best if use_best else self.f
        for s, value in zip(self.step, series):
            if value <= target:
                return s
        return None


def _grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum())
    return total**0.5


@torch.no_grad()
def _update_running_average(
    avg: list[torch.Tensor], model: nn.Module, count: int
) -> None:
    """Polyak averaging: ``avg <- avg + (w - avg) / count``, numerically stable."""
    for a, p in zip(avg, model.parameters()):
        a.add_((p.detach() - a) / count)


@torch.no_grad()
def _evaluate_at(
    model: nn.Module,
    params: list[torch.Tensor],
    objective: Callable[[nn.Module], torch.Tensor],
) -> float:
    """Evaluate ``objective`` at a given parameter vector, then restore the model."""
    saved = [p.detach().clone() for p in model.parameters()]
    for p, new in zip(model.parameters(), params):
        p.copy_(new)
    value = float(objective(model))
    for p, old in zip(model.parameters(), saved):
        p.copy_(old)
    return value


def train(
    model: nn.Module,
    loss_fn: LossFn,
    X: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int = 50,
    batch_size: int = 32,
    full_objective: Callable[[nn.Module], torch.Tensor] | None = None,
    eval_every: int = 10,
    seed: int = 42,
    track_average: bool = True,
) -> History:
    """Run stochastic subgradient descent and record its trajectory.

    ``loss_fn`` is the mini-batch objective that gets differentiated;
    ``full_objective`` is the full-dataset value recorded for the convergence
    plots (defaulting to ``loss_fn`` over all of ``X``). Keeping them separate
    matters: the stochastic estimate is what we descend on, but the full
    objective is what convergence is measured against.

    Batch shuffling uses a dedicated generator seeded from ``seed``, so runs are
    reproducible independently of global RNG state.
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X has {X.shape[0]} rows but y has {y.shape[0]}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    n = X.shape[0]
    batch_size = min(batch_size, n)
    if full_objective is None:
        full_objective = lambda m: loss_fn(m, X, y)  # noqa: E731

    generator = torch.Generator().manual_seed(seed)
    history = History()
    best = float("inf")
    avg_params = (
        [p.detach().clone() for p in model.parameters()] if track_average else []
    )

    step = 0
    n_avg = 0
    start = time.perf_counter()

    def record(grad_norm: float) -> None:
        nonlocal best
        with torch.no_grad():
            value = float(full_objective(model))
        best = min(best, value)
        history.step.append(step)
        history.f.append(value)
        history.f_best.append(best)
        history.f_avg.append(
            _evaluate_at(model, avg_params, full_objective)
            if track_average and n_avg > 0
            else value
        )
        history.grad_norm.append(grad_norm)
        history.lr.append(
            optimizer.current_lr()
            if hasattr(optimizer, "current_lr")
            else float(optimizer.param_groups[0]["lr"])
        )
        history.elapsed.append(time.perf_counter() - start)

    record(0.0)

    for _ in range(epochs):
        perm = torch.randperm(n, generator=generator)
        for start_idx in range(0, n, batch_size):
            idx = perm[start_idx : start_idx + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model, X[idx], y[idx])
            loss.backward()
            gnorm = _grad_norm(model)
            optimizer.step()
            step += 1

            if track_average:
                n_avg += 1
                _update_running_average(avg_params, model, n_avg)
            if step % eval_every == 0:
                record(gnorm)

    if not history.step or history.step[-1] != step:
        record(_grad_norm(model))
    return history


@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module, X: torch.Tensor, y: torch.Tensor, batch_size: int = 1024
) -> float:
    """Top-1 accuracy, batched so MNIST's test set fits comfortably in memory."""
    model.eval()
    correct = 0
    for i in range(0, X.shape[0], batch_size):
        logits = model(X[i : i + batch_size])
        correct += int((logits.argmax(dim=1) == y[i : i + batch_size]).sum())
    model.train()
    return correct / X.shape[0]
