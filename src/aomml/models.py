"""Models: a linear predictor for LASSO and a multi-layer ReLU network."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

__all__ = ["LinearModel", "ReLUMLP"]


class LinearModel(nn.Module):
    """``X @ weight + bias``, with the bias held separate so it can go unpenalised."""

    def __init__(
        self,
        n_features: int,
        fit_intercept: bool = True,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        # Zero init: the subgradient method's behaviour at exactly w = 0 is
        # precisely the non-smooth case this project is about, so starting there
        # exercises it from step one.
        self.weight = nn.Parameter(torch.zeros(n_features, dtype=dtype))
        self.bias = (
            nn.Parameter(torch.zeros((), dtype=dtype)) if fit_intercept else None
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        out = X @ self.weight
        return out if self.bias is None else out + self.bias


class ReLUMLP(nn.Module):
    """Fully-connected ReLU network.

    ReLU is non-differentiable at zero, so this is not merely a convenience --
    it is a genuinely non-smooth objective, which is why the subgradient method
    is the right tool for it rather than an approximation.
    """

    def __init__(
        self,
        dims: Sequence[int] = (784, 128, 10),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if len(dims) < 2:
            raise ValueError(f"need at least an input and output dim, got {dims}")

        layers: list[nn.Module] = []
        for i, (fan_in, fan_out) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(fan_in, fan_out, dtype=dtype))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.dims = tuple(dims)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)
