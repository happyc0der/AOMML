import pytest
import torch

from aomml.data import make_sparse_regression
from aomml.objectives import lasso_objective


@pytest.fixture(scope="session")
def synth():
    """Small well-conditioned synthetic LASSO problem."""
    return make_sparse_regression(n=200, d=50, k=5, noise_std=0.1, seed=42)


@pytest.fixture
def lam():
    return 0.02


@pytest.fixture
def lasso_loss(lam):
    def loss(model, Xb, yb):
        return lasso_objective(model.weight, Xb, yb, lam, model.bias)
    return loss


def sklearn_reference(data, lam):
    """Coefficients and optimal value from scikit-learn's coordinate descent."""
    from sklearn.linear_model import Lasso

    fit = Lasso(alpha=lam, fit_intercept=True, max_iter=200_000, tol=1e-14)
    fit.fit(data.X.numpy(), data.y.numpy())
    w = torch.tensor(fit.coef_, dtype=torch.float64)
    b = torch.tensor(fit.intercept_, dtype=torch.float64)
    return w, b, float(lasso_objective(w, data.X, data.y, lam, b))
