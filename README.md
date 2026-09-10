# AOMML — Stochastic Subgradient Descent, with and without Momentum

Course project implementing **stochastic subgradient descent (SSGD)** and **SSGD with heavy-ball
momentum**, applied to two non-smooth problems: **LASSO regression** and a **multi-layer ReLU
network**. Each is studied on synthetic data with known ground truth, then on real data
(California housing, MNIST).

## Quickstart

```bash
uv sync --group dev
uv run pytest                 # 87 tests
uv run ruff check .           # lint
uv run jupyter lab Project.ipynb
```

To re-execute the notebook end to end:

```bash
uv run jupyter nbconvert --execute --to notebook --inplace Project.ipynb
```

## Layout

```
src/aomml/
  optimizers.py   SSGD and SSGDMomentum as torch.optim.Optimizer subclasses
  objectives.py   LASSO objective + closed-form analytic subgradient (test oracle)
  train.py        training loop; tracks raw, best and Polyak-averaged iterates
  data.py         synthetic sparse regression, IDX reader, California housing
  models.py       LinearModel, ReLUMLP
  utils.py        seeding, device selection, Standardizer
tests/            87 tests — see "Verification" below
Project.ipynb     experimental narrative (E1–E7)
data/             MNIST IDX files, cached California housing
```

## Method

$$w_{k+1} = w_k - \alpha_k g_k, \qquad g_k \in \partial f_{i_k}(w_k)$$

with momentum $v_{k+1} = \beta v_k + g_k$, $w_{k+1} = w_k - \alpha_k v_{k+1}$, and step-size
schedules `constant`, `inv_sqrt` ($\alpha_0/\sqrt{k+1}$, the default) and `inv` ($\alpha_0/(k+1)$).

Three design points are worth knowing before reading the code:

- **The subgradient method is not a descent method.** $f(w_k)$ is non-monotone; the guarantees are
  for $\min_{j\le k} f(w_j)$ and the averaged iterate. `History` tracks all three, which is why a
  correct run can *look* unstable if you plot only the raw iterate.
- **Constant steps do not converge**, they reach a neighbourhood of $f^\star$ and hover.
  Convergence needs $\alpha_k\to 0$ with $\sum\alpha_k=\infty$.
- **Two momentum formulations.** The buffer form ($v\leftarrow\beta v+g$) and Polyak's difference
  form ($w\leftarrow w-\alpha g+\beta(w_k-w_{k-1})$) are identical at constant step but diverge
  once $\alpha_k$ decays. Both are implemented (`variant="buffer"` / `"difference"`).

## Verification

Correctness is asserted against **external references**, not by inspecting loss curves:

| Check | Evidence |
|---|---|
| Objective and gradients | autograd matches the closed-form subgradient to $10^{-12}$ |
| Non-smooth point $w=0$ | returned $g$ verified to lie in the subdifferential $[-\lambda,\lambda]$ |
| Optimizer solves LASSO | reaches scikit-learn's optimum within $10^{-3}$; coefficients within $10^{-2}$ |
| Support recovery | true support of the synthetic problem fully recovered |
| Step-size theory | constant step provably stalls (4× budget → no improvement); $1/\sqrt{k}$ keeps descending |
| Momentum | see the note below — the naive comparison overstates it by 8 orders of magnitude |
| Network wiring | MLP drives loss to ~0 on 32 samples |
| Update rules | each recurrence checked step-by-step against hand-computed values |

`tests/test_regressions.py` additionally guards every defect listed below, so a regression
reintroduces a failing test rather than a silent behaviour change.

### The momentum result, stated carefully

Comparing $\beta$ values at a fixed $\alpha_0$ is not a controlled experiment. The momentum buffer
accumulates to $v \approx g/(1-\beta)$, so the *effective* step is $\alpha_0/(1-\beta)$ — at
$\beta = 0.99$ that is a step $100\times$ larger than at $\beta = 0$. Running it both ways on
$\kappa(X) = 200$:

| | $\beta = 0$ | $\beta = 0.99$ | apparent speedup |
|---|---|---|---|
| fixed $\alpha_0 = 1$ | 2.29e-4 | 2.46e-13 | **9.3e8×** |
| effective step matched to 1 | 2.29e-4 | 2.18e-4 | **1.05×** |

At a matched step, momentum buys essentially nothing per iteration. What it does buy is
**stability**: plain SSGD diverges above an effective step of 1.5, while $\beta = 0.99$ remains
stable to an effective step of 300. That is the real mechanism — heavy-ball improves the
condition-number dependence from $\kappa$ to $\sqrt{\kappa}$ by enabling a larger stable step, not
by making each step individually smarter. E4 in the notebook runs both arms and the stability sweep.

### Evaluation hygiene

MNIST uses a 54k/6k/10k train/validation/test split; the optimiser is selected on validation
accuracy and the test set is read once. Splits go through `data.train_test_split`, which shuffles —
California housing is stored in geographic order, so a slice split shifts mean latitude by ~1.9°
between halves and silently evaluates on a different population than it trained on.

## What was wrong with the original notebook

The repository previously contained a 7-cell notebook that stated an intent and stopped: no
optimizer, no LASSO objective, no network, no training loop, no evaluation. The cells that did
exist had these defects:

| # | Defect | Consequence |
|---|---|---|
| 1 | `.cuda()` hardcoded on all tensors | hard crash on any non-NVIDIA machine |
| 2 | `device` computed then never used | the one correct line was dead code |
| 3 | `torch` never seeded | network init not reproducible, despite a cell devoted to seeding |
| 4 | no feature standardization | **iterates diverge to NaN** on real data (demonstrated in E6) |
| 5 | no intercept, `y` not centred | forces the response mean into penalised coefficients |
| 6 | Boston fetched over HTTP at runtime | **that URL now returns HTTP 403** — the cell cannot run at all |
| 7 | labels cast to `float32` | unusable by `cross_entropy`, which needs `int64` |
| 8 | `idx2numpy` imported, declared nowhere | no dependency manifest existed |
| 9 | MNIST never normalized | raw uint8 0–255 saturates a ReLU network |

Defect 6 is why this project uses **California housing** rather than Boston: the CMU host now
refuses the request, and the dataset was removed from scikit-learn in 1.2 because its `B` feature
encodes a racist assumption about neighbourhood composition. `load_boston()` is retained as an
explicit error that explains both. `idx2numpy` was dropped entirely — the IDX format is a short
`numpy.frombuffer` read (`data.read_idx`).

## A known limitation

Plain subgradient descent does **not** produce exact zeros: coefficients approach zero without
landing on it, so support recovery requires thresholding. Coordinate descent and proximal methods
(ISTA/FISTA) apply a soft-threshold and do give exact sparsity. Adding a proximal variant is the
natural extension of this work.

## Regenerating the notebook

`Project.ipynb` can be edited directly in Jupyter. It was scaffolded from
`tools/build_notebook.py`, which is kept in the repo so the narrative and code cells can be
regenerated from one reviewable file instead of hand-edited as JSON:

```bash
uv run python tools/build_notebook.py
uv run jupyter nbconvert --execute --to notebook --inplace Project.ipynb
```
