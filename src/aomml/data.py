"""Datasets: synthetic sparse regression, MNIST, and Boston housing.

The synthetic generator is the backbone of correctness checking -- it produces a
known sparse ``w_true``, so support recovery and coefficient error are testable
quantities rather than something inferred from the shape of a loss curve.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

__all__ = [
    "SparseRegressionData",
    "make_sparse_regression",
    "read_idx",
    "load_mnist",
    "load_boston",
    "load_california_housing",
    "DATA_DIR",
]

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# IDX type codes -> numpy dtypes, per the format spec used by the MNIST files.
_IDX_DTYPES = {
    0x08: np.dtype(np.uint8),
    0x09: np.dtype(np.int8),
    0x0B: np.dtype(">i2"),
    0x0C: np.dtype(">i4"),
    0x0D: np.dtype(">f4"),
    0x0E: np.dtype(">f8"),
}


@dataclass
class SparseRegressionData:
    """A synthetic LASSO problem with a known ground-truth solution."""

    X: torch.Tensor
    y: torch.Tensor
    w_true: torch.Tensor

    @property
    def support(self) -> torch.Tensor:
        """Indices of the truly non-zero coefficients."""
        return torch.nonzero(self.w_true, as_tuple=False).squeeze(-1)


def make_sparse_regression(
    n: int = 200,
    d: int = 50,
    k: int = 5,
    noise_std: float = 0.1,
    condition_number: float = 1.0,
    seed: int = 42,
    dtype: torch.dtype = torch.float64,
) -> SparseRegressionData:
    """Generate ``y = X w_true + noise`` with exactly ``k`` non-zero coefficients.

    ``condition_number > 1`` rescales the design's singular values geometrically
    to produce an ill-conditioned problem -- this is what makes the momentum
    comparison meaningful, since heavy-ball's advantage over plain subgradient
    descent only shows up when the problem is poorly conditioned.
    """
    if k > d:
        raise ValueError(f"cannot place {k} non-zeros in {d} coefficients")

    gen = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=gen, dtype=dtype)

    if condition_number != 1.0:
        if condition_number < 1.0:
            raise ValueError("condition_number must be >= 1")
        # Reshape the spectrum without disturbing the singular vectors.
        U, _, Vh = torch.linalg.svd(X, full_matrices=False)
        r = min(n, d)
        s = torch.logspace(
            0, -np.log10(condition_number), r, dtype=dtype
        ) * np.sqrt(n)
        X = U @ torch.diag(s) @ Vh

    w_true = torch.zeros(d, dtype=dtype)
    idx = torch.randperm(d, generator=gen)[:k]
    # Keep magnitudes away from zero so the true support is unambiguous.
    signs = torch.where(
        torch.rand(k, generator=gen, dtype=dtype) < 0.5, -1.0, 1.0
    ).to(dtype)
    w_true[idx] = signs * (1.0 + torch.rand(k, generator=gen, dtype=dtype))

    y = X @ w_true + noise_std * torch.randn(n, generator=gen, dtype=dtype)
    return SparseRegressionData(X=X, y=y, w_true=w_true)


def read_idx(path: str | Path) -> np.ndarray:
    """Read an IDX file into a numpy array.

    Replaces the ``idx2numpy`` dependency the original notebook imported without
    ever declaring it. The format is a 4-byte magic number (two zero bytes, a
    dtype code, a dimension count) followed by big-endian int32 dimensions and
    then the raw payload.
    """
    path = Path(path)
    raw = path.read_bytes()

    if len(raw) < 4:
        raise ValueError(f"{path.name} is too short to be an IDX file")

    zero_a, zero_b, type_code, ndim = struct.unpack(">BBBB", raw[:4])
    if zero_a != 0 or zero_b != 0:
        raise ValueError(f"{path.name}: bad IDX magic number")
    if type_code not in _IDX_DTYPES:
        raise ValueError(f"{path.name}: unknown IDX type code 0x{type_code:02X}")

    header_end = 4 + 4 * ndim
    shape = struct.unpack(f">{ndim}I", raw[4:header_end])
    dtype = _IDX_DTYPES[type_code]

    array = np.frombuffer(raw, dtype=dtype, offset=header_end)
    expected = int(np.prod(shape)) if shape else 1
    if array.size != expected:
        raise ValueError(
            f"{path.name}: expected {expected} elements for shape {shape}, "
            f"found {array.size}"
        )
    # Cast away the big-endian byte order so downstream torch conversion works.
    return array.reshape(shape).astype(dtype.newbyteorder("="))


def load_mnist(
    root: str | Path = DATA_DIR,
    flatten: bool = True,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Load MNIST from the committed IDX files.

    Images are scaled to ``[0, 1]`` (the original notebook fed raw uint8 0-255,
    which saturates a ReLU network) and labels are ``int64``, which is what
    ``cross_entropy`` requires -- the original cast them to float32.
    """
    root = Path(root)
    files = {
        "X_train": "train-images.idx3-ubyte",
        "y_train": "train-labels.idx1-ubyte",
        "X_test": "t10k-images.idx3-ubyte",
        "y_test": "t10k-labels.idx1-ubyte",
    }
    missing = [f for f in files.values() if not (root / f).exists()]
    if missing:
        raise FileNotFoundError(f"MNIST files not found in {root}: {missing}")

    out: dict[str, torch.Tensor] = {}
    for key, fname in files.items():
        arr = read_idx(root / fname)
        if key.startswith("X"):
            t = torch.from_numpy(arr).to(dtype) / 255.0
            out[key] = t.reshape(t.shape[0], -1) if flatten else t
        else:
            out[key] = torch.from_numpy(arr).to(torch.int64)
    return out


BOSTON_URL = "http://lib.stat.cmu.edu/datasets/boston"


def load_boston(*_args, **_kwargs):
    """Deprecated and no longer loadable. Use :func:`load_california_housing`.

    The original notebook fetched Boston housing from lib.stat.cmu.edu at
    runtime. That host now returns **HTTP 403**, so the original data-loading
    cell cannot execute at all today -- which is precisely why a networked
    dataset fetch does not belong in a reproducible project.

    The dataset was also removed from scikit-learn in 1.2 because its ``B``
    feature encodes a racist assumption about neighbourhood composition.
    California housing is the canonical replacement for this regression task.
    """
    raise RuntimeError(
        f"Boston housing is unavailable: {BOSTON_URL} returns HTTP 403, and the "
        "dataset was removed from scikit-learn in 1.2 over its 'B' feature. "
        "Use load_california_housing() instead."
    )


def load_california_housing(
    cache: str | Path = DATA_DIR / "california.npz",
    dtype: torch.dtype = torch.float64,
    n_samples: int | None = None,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Load California housing, caching to disk so reruns need no network.

    Replaces the original notebook's Boston fetch (see :func:`load_boston`).
    Features span very different scales -- median income in the single digits
    against population in the thousands -- so this is a real test of whether
    standardisation is being applied; without it a single penalty ``lambda``
    is meaningless across coordinates and the subgradient steps diverge.

    ``n_samples`` optionally subsamples for faster experiments.
    """
    cache = Path(cache)
    if cache.exists():
        blob = np.load(cache, allow_pickle=True)
        data, target = blob["data"], blob["target"]
        names = [str(v) for v in blob["names"]]
    else:
        from sklearn.datasets import fetch_california_housing

        bundle = fetch_california_housing()
        data, target = bundle.data, bundle.target
        names = list(bundle.feature_names)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, data=data, target=target, names=np.array(names))

    if n_samples is not None and n_samples < data.shape[0]:
        rng = np.random.default_rng(seed)
        idx = rng.choice(data.shape[0], size=n_samples, replace=False)
        data, target = data[idx], target[idx]

    return (
        torch.from_numpy(np.ascontiguousarray(data)).to(dtype),
        torch.from_numpy(np.ascontiguousarray(target)).to(dtype),
        names,
    )
