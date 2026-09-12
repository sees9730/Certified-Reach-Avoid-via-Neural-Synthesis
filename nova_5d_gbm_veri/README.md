# 5D GBM verification: bound training in one small implementation

`main.py` contains the model, interval bounds, training, adaptive refinement,
and final checker (about 300 lines). Plotting is isolated in `plotting.py`. Everything is local to this folder;
there are no imports from the surrounding repository, pretrained weights,
linear-programming solver, or problem-specific proof shortcuts.

## Run

From the repository root, using its existing Python environment:

```sh
./.venv/bin/python -m nova_5d_gbm_veri train
./.venv/bin/python -m nova_5d_gbm_veri verify
./.venv/bin/python -m unittest nova_5d_gbm_veri.test_nova_5d_gbm_veri -v
```

Dependencies are managed by the shared [root requirements.txt](../requirements.txt).
From the repository root, install them with `python -m pip install -r requirements.txt`.
Direct execution also works: `python nova_5d_gbm_veri/main.py train`.
The source folder remains self-contained; a standalone copy uses the same installed
dependencies and does not need its own requirements file.

The default input is `nova_5d_gbm_veri/5d_gbm.json`. It contains the exact original
5D GBM dynamics, multiplicative noise 0.2*x_i, domain **[-100,100]^5**,
initial/unsafe/goal boxes, beta_ra=20, and epsilon=0.0001. The algorithm's
model and optimizer settings are new; the mathematical problem is unchanged.

## Certificate function model (this is the V)

There is no controller here: `5d_gbm.json`'s drift/diffusion are fixed and
uncontrolled, so synthesis only ever trains `V`.

`V = sum_j w_j * phi_j(x)`, with trainable nonnegative weights `w_j` and a
fixed feature basis `phi_j`, all built once from the problem geometry, not
loaded:

- `z_i = (x_i - center_i) / scale_i`, normalized by the domain and the goal
  center.
- `q = (sum_i z_i²) / radius²`, radius chosen so `q<=1` roughly covers the
  initial region.
- Features: `q¹` through `q^degree` (default `degree=8`), plus `z_i²` and
  `(z_i ± z_j)²` for every coordinate pair — a fixed polynomial basis, not a
  neural network.

Every feature is a square or a positive power of a sum of squares, and
`w_j >= 0` is reprojected after every optimizer step, so **V >= 0 on the
whole domain by construction** — this is never something training or
verification needs to check.

`Bounds.__init__` differentiates each feature through the SDE's full Itô
generator (`sp.diff`/`sp.expand`, including mixed diffusion terms) using
`sympy`, and stores every resulting polynomial's coefficients as exact
`Fraction`s. `Bounds.bounds(cell)` then bounds each monomial on a box with
exact rational interval arithmetic and caches the result — so **every
feature/generator bound used anywhere below is already exact**, independent
of the (float64) weights. The GBM is tractable this way because the exact
symbolic combination cancels its skew drift terms, producing tight radial
bounds; there's no separate GBM theorem or hardcoded generator inequality.

## Training (how training reaches a SAT-valid certificate)

1. **Initialize.** All weights start at `0.005`. The partition starts with
   one box per initial/unsafe region and a single generator cell covering
   the whole domain.
2. **Loss from cached exact bounds.** The (already-exact) per-cell feature
   bounds are cast to float64 once per partition and combined with the
   current weights for a projected-Adam loss step:

   ```text
   sum relu(V_initial_upper - 0.99)
   + sum relu(1.01*beta_ra - V_unsafe_lower) / beta_ra
   + sum_active relu(GV_upper + 1.1*epsilon) / epsilon
   ```

   A generator cell is active if it is not wholly inside the goal and its
   `V` lower bound is below `beta_ra`. `w>=0` is reclamped after every step.
3. **Checkpoint (every `--refine-every` steps, on zero loss, or the final
   epoch): call the real exact-rational `verify()`.** The current weights
   are converted to `Fraction` exactly (`Q(float(v))` — lossless, since a
   Python float already *is* a binary64 rational) and passed into the same
   `verify()` used by the standalone `verify` command, over the same cached
   exact cell bounds. Unlike a float64 approximation, this is already a
   fully rigorous SAT/UNKNOWN verdict:
   `V_initial_upper <= 1`, `V_unsafe_lower >= beta_ra`, and every generator
   cell satisfies `inside_goal OR V_lower >= beta_ra OR GV_upper <= -epsilon`.
   - **If SAT, stop immediately** — because the check itself was exact, this
     result is already a genuine proof, not a training heuristic to be
     upgraded later.
   - Otherwise, **refine**: score failing cells by normalized bound
     violation, split up to `--batch` of them (a goal-face split for
     generator cells crossing the goal, otherwise the longest normalized
     axis), replacing each parent by both children so no region is lost, up
     to `--max-cells`, and continue.
4. **Termination.** The loop above stops with `SAT` as soon as a checkpoint's
   exact check passes; otherwise it stops with `UNKNOWN` once `--seconds`
   wall-clock or `--epochs` is exhausted. `UNKNOWN` means the budget ran out,
   not that the problem is infeasible — no algorithm here guarantees success
   for an arbitrary changed problem.

Because `train`'s own stopping check already uses the exact rational
`verify()`, a `train` run that reports SAT needs no further step to be
trusted. The standalone `verify` command is for auditing a *saved*
`certificate.json` independently later: it reconstructs the model from
scratch, replays its recorded splits, and recomputes every bound, trusting
nothing cached from the original run.

## Inspect the result

`nova_5d_gbm_veri/results/` contains:

- `training.csv`: bound-loss components and cell counts over training.
- `certificate.json`: every learned weight, feature formula, normalization,
  input problem, training time, and the complete sequence of cell splits.
- `verification.json`: the exact domain, worst bounds, and each final cell's
  bounds, verdict, and reason for acceptance.

The generator condition is checked on `{x in domain: V(x)<beta_ra} \ goal`.
V>=0 holds globally. The domain is never enlarged. No extra requirement
that V>=beta_ra on the entire domain boundary is imposed: that is not one
of the four supplied conditions.

`verify` starts from the original region boxes, validates/replays every split,
regenerates the model, and recomputes every bound. It does not trust the saved
SAT label, saved bounds, or optimizer. Training and verification both return
exit code 0 for SAT, 1 for UNKNOWN, and 2 for invalid input.

## Visualize after verification

`python -m nova_5d_gbm_veri verify` automatically saves `results/plots/V_GV.pdf` and PNG
images. For the 5D example the PDF has separate pages through the goal,
initial, and unsafe box centers, plotting x1 against x2. Only boxes that
actually intersect each slice are outlined. Axes always stay within the input
domain: **[-100,100]** here. The figure titles show every fixed coordinate.

The plots include labeled value contours, plus V=1, V=beta_ra, and GV=-epsilon wherever they cross
the sampled grid. A missing beta contour is explicitly noted; the domain is
never enlarged to make a contour appear. Color scales are logarithmic away
from zero. These are pointwise V/GV visualizations, not interval proof bounds.

Choose a custom slice (unlisted fixed coordinates use the goal center):

```sh
./.venv/bin/python -m nova_5d_gbm_veri verify --axes x1 x3 --slice x2=-50 x4=50 x5=50
```

Use `--points 401` for a finer plot grid, or `--no-plots` to run verification
alone. A one-dimensional problem gets line plots. `plots/slices.json` records
the chosen coordinates, intersecting regions, and sampled value ranges.
Plotting requires Matplotlib, included in the root `requirements.txt`.

## Change the problem or budget

Copy `5d_gbm.json` and change its domain, polynomial drift/diffusion expressions,
region boxes, beta_ra, or epsilon. Regions are lists of boxes; the diffusion
is an n-by-m matrix multiplying m independent Brownian components. All region
boxes must lie inside the domain. Nonpolynomial dynamics are unsupported.

```sh
python nova_5d_gbm_veri/main.py train --problem my_problem.json --output nova_5d_gbm_veri/my_run \
  --degree 8 --epochs 5000 --lr 0.01 --refine-every 25 --batch 4 \
  --max-cells 4096 --seconds 120
python nova_5d_gbm_veri/main.py verify --problem my_problem.json --output nova_5d_gbm_veri/my_run
```

`degree` selects the highest radial power (polynomial degree is twice that).
The search family is finite and interval bounds can be conservative. Increasing
degree/cell/iteration budgets may help but is not a completeness guarantee.
The time budget is checked at refinement/check points and excludes model setup;
the reported total includes setup but not process import overhead. Reusing an
output directory replaces its previous results.
