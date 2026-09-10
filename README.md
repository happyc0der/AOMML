# AOMML — Stochastic Subgradient Descent, with and without Momentum

Course project implementing **stochastic subgradient descent (SSGD)** and **SSGD with heavy-ball
momentum** from the ground up, applied to two genuinely non-smooth problems: **LASSO regression**
(the $\ell_1$ penalty is non-differentiable at zero) and a **multi-layer ReLU network** (the ReLU
kink). Each is studied on synthetic data with a known ground truth, then on real data.

## Requirements

- **Python 3.12**
- macOS/Linux/Windows. Runs on CPU; uses Apple MPS automatically when available.
- No network needed — MNIST and the California housing cache are committed.

## Running

```bash
uv sync --group dev        # recommended: exact pins from uv.lock
uv run pytest              # 89 tests, ~9s
uv run jupyter lab Project.ipynb
```

Without `uv`:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
pytest
```

Re-execute the notebook end to end (~3 min):

```bash
uv run jupyter nbconvert --execute --to notebook --inplace Project.ipynb
```

## What was done

The repository previously held a single 7-cell notebook that stated an intent and stopped — no
optimizer, no objective, no network, no training loop, no evaluation. The cells that did exist had
nine defects, two of them fatal: `.cuda()` was hardcoded (a crash on any non-NVIDIA machine), and
the Boston housing fetch returns **HTTP 403** today, so the notebook could not run at all.

Built from that starting point:

- **`optimizers.py`** — `SSGD` and `SSGDMomentum` as real `torch.optim.Optimizer` subclasses, with
  `constant` / `inv_sqrt` / `inv` step schedules, optional Nesterov, and both the buffer and
  Polyak difference formulations of momentum.
- **`objectives.py`** — LASSO with an unpenalised intercept, plus a closed-form analytic
  subgradient kept purely as a test oracle against autograd.
- **`train.py`** — training loop tracking the raw, running-best and Polyak-averaged iterates
  separately, because the subgradient method is *not* a descent method.
- **`data.py`, `models.py`, `utils.py`** — synthetic sparse regression with known `w_true`, an IDX
  reader (replacing the undeclared `idx2numpy`), shuffled splits, seeding, device selection, scaling.
- **89 tests** and a **CI** workflow (lint, format, tests, full notebook execution).

Fixed along the way: hardcoded `.cuda()`, unused device variable, `torch` never seeded, no feature
standardization, no intercept, labels cast to `float32`, unnormalized `uint8` images, the dead
Boston URL, and the undeclared dependency. Each has a regression test in `tests/test_regressions.py`.

## What was achieved

Correctness is anchored to **external references**, not to loss curves looking plausible:

| Check | Result |
|---|---|
| Autograd vs. closed-form subgradient | agrees to `4e-16` |
| Subgradient at the non-smooth point `w=0` | verified valid in $[-\lambda,\lambda]$, all coordinates |
| LASSO optimum vs. scikit-learn | gap `7.4e-5`, coefficients within `1.5e-3` |
| Support recovery | true support recovered at 10/10 tested $\lambda$ |
| Update rules | each recurrence matched step-by-step, and against `torch.optim.SGD` to `1e-14` |
| MNIST (784→128→10, 5 epochs) | 91.4% test (SSGD), 92.7% (momentum), selected on a held-out validation split |

Three results worth stating:

1. **The subgradient method is not a descent method.** $f(w_k)$ rose on **44%** of iterations while
   converging normally. Plotting only the raw iterate makes a correct implementation look broken.
2. **A constant step does not converge.** It reaches a neighbourhood of $f^\star$ and stops:
   quadrupling the budget changed its gap by less than $10^{-6}$ relative (`5.4616` → `5.4616`),
   while $\alpha_0/\sqrt{k+1}$ kept descending.
3. **Momentum's benefit is stability, not a better step.** Comparing $\beta$ at fixed $\alpha_0$
   suggests a `9.3e8×` speedup — but that silently varies the effective step $\alpha_0/(1-\beta)$.
   With the effective step *matched*, the advantage collapses to `1.05×`. What momentum actually
   buys is headroom: plain SSGD diverges above an effective step of `1.5`, while $\beta=0.99$
   stays stable to `300`. That is the $\kappa \to \sqrt{\kappa}$ result as a mechanism rather than
   a slogan.

## Layout

```
src/aomml/      optimizers, objectives, models, training loop, data, utils
tests/          89 tests: math, update rules, convergence, defect regressions
tools/          build_notebook.py — regenerates Project.ipynb
Project.ipynb   experimental narrative (E1–E7)
data/           MNIST IDX files, cached California housing
```

## Known limitation

Plain subgradient descent does **not** produce exact zeros — coefficients approach zero without
landing on it (0 of 50 exactly zero, against scikit-learn's 45), so support recovery requires
thresholding. Proximal methods (ISTA/FISTA) apply a soft-threshold and do give exact sparsity;
adding one is the natural extension of this work.

## Note on the dataset

Boston housing was replaced by California housing: the source URL now returns HTTP 403, and the
dataset was removed from scikit-learn 1.2 because its `B` feature encodes a racist assumption about
neighbourhood composition. `load_boston()` is retained as an explicit error explaining both.
