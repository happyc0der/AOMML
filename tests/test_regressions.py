"""Guards against the specific defects found in the original notebook.

Each test here maps to a numbered defect in the project plan, so a regression
reintroduces a failing test rather than a silent behaviour change.
"""

import ast
from pathlib import Path

import pytest
import torch

from aomml.data import make_sparse_regression
from aomml.models import LinearModel, ReLUMLP
from aomml.objectives import lasso_objective
from aomml.optimizers import SSGD
from aomml.train import train
from aomml.utils import Standardizer, get_device, seed_everything

SRC = Path(__file__).resolve().parents[1] / "src" / "aomml"


def test_no_hardcoded_cuda_calls():
    """Defect #1: `.cuda()` hardcoded, a hard crash on any non-NVIDIA machine.

    Parsed from the syntax tree rather than grepped, so prose in docstrings that
    *describes* the defect does not trip the guard that forbids it.
    """
    offenders = []
    for path in SRC.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "cuda"
            ):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, f"hardcoded .cuda() call found at {offenders}"


def test_device_selection_returns_usable_device():
    """Defect #2: the original computed `device` and then ignored it."""
    dev = get_device(dtype=torch.float32)
    assert dev.type in {"cpu", "mps"}
    torch.zeros(2, device=dev)  # must not raise


def test_float64_never_routed_to_mps():
    """MPS has no float64 kernels; silently downcasting would break the
    tolerance of the scikit-learn comparison tests."""
    assert get_device(prefer="auto", dtype=torch.float64).type == "cpu"
    assert get_device(prefer="mps", dtype=torch.float64).type == "cpu"


def test_seed_everything_makes_torch_reproducible():
    """Defect #3: torch was never seeded, so network init was not reproducible
    despite the notebook's cell claiming to set seeds for reproducibility."""
    seed_everything(7)
    a = ReLUMLP((10, 8, 3)).net[0].weight.detach().clone()
    seed_everything(7)
    b = ReLUMLP((10, 8, 3)).net[0].weight.detach().clone()
    assert torch.equal(a, b)

    seed_everything(8)
    c = ReLUMLP((10, 8, 3)).net[0].weight.detach().clone()
    assert not torch.equal(a, c)


def test_standardizer_refuses_transform_before_fit():
    """Defect #4: no scaling at all. The guard makes leakage impossible to
    introduce by accident, since test data cannot be transformed by its own
    statistics without an explicit fit."""
    with pytest.raises(RuntimeError, match="before fit"):
        Standardizer().transform(torch.randn(4, 3))


def test_standardizer_uses_training_statistics_only():
    train_X = torch.randn(100, 5, dtype=torch.float64) * 10 + 3
    test_X = torch.randn(40, 5, dtype=torch.float64) * 10 + 50  # deliberately shifted

    scaler = Standardizer().fit(train_X)
    scaled_test = scaler.transform(test_X)

    # If test statistics had leaked in, the test set would be centred at zero.
    assert scaled_test.mean(dim=0).abs().min() > 1.0
    # Training data, by contrast, is centred.
    assert scaler.transform(train_X).mean(dim=0).abs().max() < 1e-10


def test_standardizer_handles_constant_columns():
    X = torch.ones(20, 3, dtype=torch.float64)
    X[:, 1] = torch.randn(20, dtype=torch.float64)
    out = Standardizer().fit_transform(X)
    assert torch.isfinite(out).all()


def test_standardizer_rejects_wrong_feature_count():
    scaler = Standardizer().fit(torch.randn(10, 4))
    with pytest.raises(ValueError):
        scaler.transform(torch.randn(10, 5))


def test_training_is_deterministic():
    """Same seed, same trajectory -- twice."""
    data = make_sparse_regression(n=120, d=20, k=3, seed=9)
    loss = lambda m, Xb, yb: lasso_objective(m.weight, Xb, yb, 0.02, m.bias)  # noqa: E731

    def run():
        seed_everything(0)
        model = LinearModel(20)
        return train(
            model, loss, data.X, data.y,
            SSGD(model.parameters(), lr=0.1, schedule="inv_sqrt"),
            epochs=20, batch_size=16, eval_every=5, seed=3,
        )

    a, b = run(), run()
    assert a.f == b.f
    assert a.f_best == b.f_best
