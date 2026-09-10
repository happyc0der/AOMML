"""Reproducibility, device selection, and feature scaling.

Fixes three defects carried over from the original notebook:
  * seeds were set for ``random``/``numpy`` but never for ``torch``, so anything
    involving network initialisation was not actually reproducible;
  * ``.cuda()`` was hardcoded, which is a hard crash on any machine without an
    NVIDIA device;
  * features were never standardised, which for a dataset like Boston (scales
    spanning ~0.4 to ~700) makes subgradient descent diverge and makes a single
    penalty ``lambda`` meaningless across coordinates.
"""

from __future__ import annotations

import random

import numpy as np
import torch

__all__ = ["seed_everything", "get_device", "Standardizer"]


def seed_everything(seed: int = 42) -> None:
    """Seed every RNG this project draws from.

    The original notebook seeded ``random`` and ``numpy`` only, so torch's
    parameter initialisation stayed random run to run.
    """
    random.seed(seed)
    # Legacy global seed on purpose: scikit-learn and other libraries here still
    # draw from numpy's global RNG, so a Generator would not cover them.
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device(prefer: str = "auto", dtype: torch.dtype = torch.float32) -> torch.device:
    """Pick a device, respecting what the requested dtype can actually run on.

    ``prefer="auto"`` selects MPS when available, otherwise CPU. float64 is not
    supported by the MPS backend, so a float64 request always resolves to CPU --
    this is what lets the LASSO experiments run in double precision and be
    compared against scikit-learn at a tight tolerance.
    """
    if prefer not in {"auto", "cpu", "mps"}:
        raise ValueError(f"unknown device preference {prefer!r}")

    if dtype == torch.float64:
        # MPS has no float64 kernels; silently running float32 here would break
        # the tolerance of the sklearn comparison tests.
        return torch.device("cpu")

    if prefer == "cpu":
        return torch.device("cpu")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if prefer == "mps":
        raise RuntimeError("MPS requested but not available on this machine")

    return torch.device("cpu")


class Standardizer:
    """Zero-mean unit-variance scaling, fitted on training data only.

    ``transform`` refuses to run before ``fit``, so applying test data with test
    statistics -- the usual silent leakage bug -- cannot happen by accident.
    """

    def __init__(self, eps: float = 1e-12) -> None:
        self.eps = eps
        self.mean_: torch.Tensor | None = None
        self.std_: torch.Tensor | None = None

    @property
    def fitted(self) -> bool:
        return self.mean_ is not None

    def fit(self, X: torch.Tensor) -> Standardizer:
        if X.ndim != 2:
            raise ValueError(f"expected a 2-D design matrix, got shape {tuple(X.shape)}")
        self.mean_ = X.mean(dim=0)
        std = X.std(dim=0, unbiased=False)
        # Constant columns would divide by zero; leave them untouched instead.
        self.std_ = torch.where(std < self.eps, torch.ones_like(std), std)
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        if not self.fitted:
            raise RuntimeError(
                "Standardizer.transform called before fit -- fit on the training "
                "split and reuse those statistics for test data"
            )
        if X.shape[1] != self.mean_.shape[0]:
            raise ValueError(f"expected {self.mean_.shape[0]} features, got {X.shape[1]}")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        return self.fit(X).transform(X)
