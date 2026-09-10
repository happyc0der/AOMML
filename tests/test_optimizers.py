"""The update rules: does each optimizer apply exactly the recurrence it claims?"""

import math

import pytest
import torch

from aomml.optimizers import SSGD, SSGDMomentum, schedule_factor


@pytest.mark.parametrize(
    "schedule,k,expected",
    [
        ("constant", 0, 1.0), ("constant", 99, 1.0),
        ("inv_sqrt", 0, 1.0), ("inv_sqrt", 1, 1 / math.sqrt(2)), ("inv_sqrt", 99, 0.1),
        ("inv", 0, 1.0), ("inv", 1, 0.5), ("inv", 99, 0.01),
    ],
)
def test_lr_schedule_values(schedule, k, expected):
    assert schedule_factor(schedule, k) == pytest.approx(expected)


def test_unknown_schedule_rejected():
    with pytest.raises(ValueError):
        schedule_factor("cosine", 0)
    with pytest.raises(ValueError):
        SSGD([torch.nn.Parameter(torch.zeros(1))], schedule="cosine")


def test_ssgd_single_step_is_exact():
    p = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float64))
    opt = SSGD([p], lr=0.1, schedule="constant")
    p.grad = torch.tensor([0.5, -1.0], dtype=torch.float64)
    opt.step()
    assert torch.allclose(p.detach(), torch.tensor([0.95, 2.1], dtype=torch.float64))


def test_ssgd_follows_decaying_schedule():
    """Step k must scale by 1/sqrt(k+1), so three unit-gradient steps move
    the iterate by exactly the partial sum of that series."""
    p = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    opt = SSGD([p], lr=1.0, schedule="inv_sqrt")
    for _ in range(3):
        p.grad = torch.ones(1, dtype=torch.float64)
        opt.step()
    expected = -sum(1 / math.sqrt(k + 1) for k in range(3))
    assert p.item() == pytest.approx(expected, abs=1e-12)


def test_momentum_recurrence_is_exact():
    """Three steps must match v <- beta*v + g ; w <- w - alpha*v exactly."""
    p = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    opt = SSGDMomentum([p], lr=0.1, beta=0.9, schedule="constant")
    v = w = 0.0
    for g in (1.0, 1.0, -2.0):
        p.grad = torch.tensor([g], dtype=torch.float64)
        opt.step()
        v = 0.9 * v + g
        w = w - 0.1 * v
        assert p.item() == pytest.approx(w, abs=1e-12)


def test_momentum_variants_agree_under_constant_step():
    """Buffer and difference forms are algebraically identical when alpha is
    fixed -- telescoping gives w_k - w_{k-1} = -alpha*v_k."""
    def run(variant):
        p = torch.nn.Parameter(torch.tensor([3.0], dtype=torch.float64))
        opt = SSGDMomentum([p], lr=0.1, beta=0.9, schedule="constant", variant=variant)
        for _ in range(25):
            p.grad = 2 * p.detach()
            opt.step()
        return p.item()

    assert run("buffer") == pytest.approx(run("difference"), abs=1e-12)


def test_momentum_variants_diverge_under_decaying_step():
    """...and are genuinely different once alpha decays, because the buffer form
    rescales the whole accumulated history by the current alpha_k."""
    def run(variant):
        p = torch.nn.Parameter(torch.tensor([3.0], dtype=torch.float64))
        opt = SSGDMomentum([p], lr=0.1, beta=0.9, schedule="inv_sqrt", variant=variant)
        for _ in range(25):
            p.grad = 2 * p.detach()
            opt.step()
        return p.item()

    assert abs(run("buffer") - run("difference")) > 1e-3


def test_buffer_variant_matches_torch_sgd():
    """Sanity anchor: at constant step our buffer form is torch.optim.SGD."""
    ours = torch.nn.Parameter(torch.tensor([3.0], dtype=torch.float64))
    torch_p = torch.nn.Parameter(torch.tensor([3.0], dtype=torch.float64))
    a = SSGDMomentum([ours], lr=0.1, beta=0.9, schedule="constant")
    b = torch.optim.SGD([torch_p], lr=0.1, momentum=0.9)
    for _ in range(15):
        ours.grad = 2 * ours.detach()
        torch_p.grad = 2 * torch_p.detach()
        a.step()
        b.step()
    assert ours.item() == pytest.approx(torch_p.item(), abs=1e-14)


@pytest.mark.parametrize("beta", [-0.1, 1.0, 1.5])
def test_invalid_beta_rejected(beta):
    with pytest.raises(ValueError):
        SSGDMomentum([torch.nn.Parameter(torch.zeros(1))], beta=beta)


def test_zero_beta_reduces_to_plain_ssgd():
    a = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
    b = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
    o1 = SSGD([a], lr=0.05, schedule="inv_sqrt")
    o2 = SSGDMomentum([b], lr=0.05, beta=0.0, schedule="inv_sqrt")
    for _ in range(20):
        a.grad = 2 * a.detach()
        b.grad = 2 * b.detach()
        o1.step()
        o2.step()
    assert a.item() == pytest.approx(b.item(), abs=1e-14)
