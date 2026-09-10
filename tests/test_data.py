"""Data loading, including the IDX reader that replaced idx2numpy."""

import struct

import numpy as np
import pytest
import torch

from aomml.data import load_mnist, make_sparse_regression, read_idx


def test_mnist_dtypes_and_ranges():
    """Images scaled to [0,1] float, labels int64 in [0,9].

    The original notebook cast labels to float32 (unusable by cross_entropy) and
    fed raw uint8 0-255 images into the network (which saturates ReLU units).
    """
    d = load_mnist()

    assert d["X_train"].dtype == torch.float32
    assert d["y_train"].dtype == torch.int64
    assert float(d["X_train"].min()) >= 0.0 and float(d["X_train"].max()) <= 1.0
    assert int(d["y_train"].min()) >= 0 and int(d["y_train"].max()) <= 9


def test_mnist_shapes():
    d = load_mnist()
    assert d["X_train"].shape == (60_000, 784)
    assert d["y_train"].shape == (60_000,)
    assert d["X_test"].shape == (10_000, 784)
    assert d["y_test"].shape == (10_000,)


def test_mnist_unflattened_is_square():
    d = load_mnist(flatten=False)
    assert d["X_train"].shape == (60_000, 28, 28)


def test_idx_reader_roundtrip(tmp_path):
    """Reader must reproduce a known payload, dimensions and all."""
    payload = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    blob = struct.pack(">BBBB", 0, 0, 0x08, 3) + struct.pack(">3I", 2, 3, 4)
    blob += payload.tobytes()
    path = tmp_path / "sample.idx3-ubyte"
    path.write_bytes(blob)

    assert np.array_equal(read_idx(path), payload)


def test_idx_reader_rejects_bad_magic(tmp_path):
    path = tmp_path / "bad.idx"
    path.write_bytes(struct.pack(">BBBB", 1, 0, 0x08, 1) + struct.pack(">I", 1) + b"\x00")
    with pytest.raises(ValueError, match="magic"):
        read_idx(path)


def test_idx_reader_rejects_truncated_payload(tmp_path):
    path = tmp_path / "short.idx"
    path.write_bytes(struct.pack(">BBBB", 0, 0, 0x08, 1) + struct.pack(">I", 10) + b"\x00\x01")
    with pytest.raises(ValueError, match="expected"):
        read_idx(path)


def test_synthetic_has_exact_sparsity():
    d = make_sparse_regression(n=100, d=30, k=4, seed=1)
    assert int((d.w_true != 0).sum()) == 4
    assert d.support.numel() == 4


def test_synthetic_signal_dominates_noise():
    """With tiny noise the response must be nearly exactly X @ w_true."""
    d = make_sparse_regression(n=100, d=20, k=3, noise_std=1e-8, seed=2)
    assert torch.allclose(d.y, d.X @ d.w_true, atol=1e-6)


@pytest.mark.parametrize("kappa", [10.0, 200.0])
def test_condition_number_is_honoured(kappa):
    d = make_sparse_regression(n=200, d=30, condition_number=kappa, seed=5)
    assert float(torch.linalg.cond(d.X)) == pytest.approx(kappa, rel=1e-6)


def test_synthetic_is_reproducible():
    a = make_sparse_regression(seed=123)
    b = make_sparse_regression(seed=123)
    c = make_sparse_regression(seed=124)
    assert torch.equal(a.X, b.X) and torch.equal(a.w_true, b.w_true)
    assert not torch.equal(a.X, c.X)


def test_too_many_nonzeros_rejected():
    with pytest.raises(ValueError):
        make_sparse_regression(d=5, k=10)


def test_load_boston_raises_with_explanation():
    """Defect #6: the CMU host now returns HTTP 403, so the original notebook's
    data cell cannot execute at all. The replacement must say so explicitly
    rather than failing with an opaque urllib error."""
    from aomml.data import load_boston

    with pytest.raises(RuntimeError, match="403"):
        load_boston()


def test_california_housing_loads_from_cache():
    from aomml.data import load_california_housing

    X, y, names = load_california_housing()
    assert X.shape[0] == y.shape[0] == 20_640
    assert X.shape[1] == len(names) == 8
    assert X.dtype == torch.float64
    assert torch.isfinite(X).all() and torch.isfinite(y).all()


def test_california_subsampling_is_reproducible():
    from aomml.data import load_california_housing

    a, ya, _ = load_california_housing(n_samples=500, seed=1)
    b, yb, _ = load_california_housing(n_samples=500, seed=1)
    c, _, _ = load_california_housing(n_samples=500, seed=2)
    assert a.shape[0] == 500
    assert torch.equal(a, b) and torch.equal(ya, yb)
    assert not torch.equal(a, c)


def test_california_features_are_badly_scaled():
    """The premise of the standardisation experiment: without scaling, one
    coordinate's gradient dwarfs another's by three orders of magnitude."""
    from aomml.data import load_california_housing

    X, _, _ = load_california_housing()
    spread = float(X.std(dim=0).max() / X.std(dim=0).min())
    assert spread > 100


def test_train_test_split_shuffles():
    """Slicing an ordered dataset evaluates on a different population than it
    trained on. California housing is stored geographically, so an unshuffled
    split shifts mean latitude by ~1.9 degrees between halves."""
    from aomml.data import load_california_housing, train_test_split

    X, y, _ = load_california_housing()
    lat = 6  # Latitude column

    n = int(0.8 * len(X))
    slice_shift = abs(float(X[:n, lat].mean() - X[n:, lat].mean()))

    Xtr, Xte, _, _ = train_test_split(X, y, test_size=0.2, seed=42)
    shuffled_shift = abs(float(Xtr[:, lat].mean() - Xte[:, lat].mean()))

    assert slice_shift > 1.0, "premise: the raw data really is ordered"
    assert shuffled_shift < 0.1, "shuffled split must not shift the distribution"


def test_train_test_split_sizes_and_partition():
    from aomml.data import train_test_split

    X = torch.arange(100, dtype=torch.float64).reshape(100, 1)
    y = torch.arange(100, dtype=torch.float64)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, seed=0)

    assert Xtr.shape[0] == ytr.shape[0] == 75
    assert Xte.shape[0] == yte.shape[0] == 25
    # Every row appears exactly once across the two halves.
    combined = torch.cat([Xtr.flatten(), Xte.flatten()]).sort().values
    assert torch.equal(combined, torch.arange(100, dtype=torch.float64))


def test_train_test_split_keeps_rows_aligned():
    """X and y must be permuted together, or every label is wrong."""
    from aomml.data import train_test_split

    X = torch.arange(60, dtype=torch.float64).reshape(60, 1)
    y = X.flatten() * 10
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, seed=1)
    assert torch.equal(ytr, Xtr.flatten() * 10)
    assert torch.equal(yte, Xte.flatten() * 10)


def test_train_test_split_is_reproducible():
    from aomml.data import train_test_split

    X = torch.randn(50, 3, dtype=torch.float64)
    y = torch.randn(50, dtype=torch.float64)
    a = train_test_split(X, y, seed=7)[0]
    b = train_test_split(X, y, seed=7)[0]
    c = train_test_split(X, y, seed=8)[0]
    assert torch.equal(a, b) and not torch.equal(a, c)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_train_test_split_rejects_invalid_test_size(bad):
    from aomml.data import train_test_split

    with pytest.raises(ValueError):
        train_test_split(torch.randn(10, 2), torch.randn(10), test_size=bad)


def test_train_test_split_rejects_mismatched_lengths():
    from aomml.data import train_test_split

    with pytest.raises(ValueError):
        train_test_split(torch.randn(10, 2), torch.randn(9))
