# XV-15 uncertain-parameter experiments

The uncertain-density-and-mass experiment is available in
[`neural_certified_uncertain_param/`](neural_certified_uncertain_param/README.md).
It uses the nominal training settings with a robust generator for air density
in **[1.16375, 1.28625] kg/m³** (±5%) and configurable mass uncertainty
(for example, **[5841, 5959] kg**, ±1% around 5900 kg). It selects the tightest
IBP bound from three equivalent robust drift formulas. The instructions below describe the
nominal baseline.

The first experiment, `neural_certified_nominal_drift/`, establishes a nominal
baseline before parameter uncertainty is introduced. It synthesizes a neural
certificate and a neural controller using the same workflow as
`examples/inv_pend_adversarial/neural_certified_nominal_drift`:
sample pretraining from scratch, adaptive bound training, final evaluation,
and plots. It uses the shared `src/` implementation.

This stage uses fixed nominal physical parameters and retains the SDE's
Brownian diffusion. It does not certify uncertain parameters. There is no
time/energy augmentation, curriculum, or checkpoint loading.

## Monte Carlo controller comparison

`run_mc.py` compares saved controllers over the Cartesian product of air-density
and mass realization modes. From the repository root:

```bash
./.venv/bin/python -u examples/xv15_uncertain/run_mc.py \
    --n-mc 500 --t-max 20 --dt 0.01 \
    --density-min 1.16375 --density-max 1.28625 \
    --mass-min 5841 --mass-max 5959
```

This explicitly selects ±5% density and ±1% mass. Without these overrides,
the intervals come from the current `config.json` (which can be edited for
other experiments). All controllers use the same evaluation physics, regions,
and uncertainty intervals. Each controller retains its saved normalization,
trim anchor, and physical action limits; changing realized mass never rescales
the commanded thrust.

The default run discovers every available `seed<N>` checkpoint under
`neural_certified_uncertain_param/`, `neural_certified_nominal_drift/`, and
`rl_sb3_ppo/`. Missing default controllers are skipped with a message.
Certified runs load `outputs/eval_bundle.pth` and the adjacent
`run_config.json`; PPO runs load the `xv15_sb3_ppo_actor_v1` export in
`outputs/rl_controller.pth`, without requiring SB3 during evaluation.
The controller labels describe training methods, not whether the saved
candidate reached bound-training SAT.

Use repeated `--controller LABEL=PATH` arguments to select a subset, individual
training seeds, or other compatible runs. Paths are relative to the current
working directory; a path may name a controller directory, seed directory,
`outputs` directory, or checkpoint file. For example:

```bash
./.venv/bin/python -u examples/xv15_uncertain/run_mc.py \
    --controller 'Robust=examples/xv15_uncertain/neural_certified_uncertain_param/seed0' \
    --controller 'Nominal=examples/xv15_uncertain/neural_certified_nominal_drift/seed0' \
    --controller 'PPO=examples/xv15_uncertain/rl_sb3_ppo/seed0' \
    --density-modes zero uniform adversarial \
    --mass-modes zero uniform adversarial \
    --mass-min 5841 --mass-max 5959 --n-mc 500
```

### Parameter realization modes

Density and mass modes are chosen independently; defaults evaluate all nine
combinations, including uniform density with adversarial mass and vice versa.

| Mode | Realization |
|---|---|
| `zero` | Nominal density or mass from `dynamics`; this means zero **uncertainty**, not zero density/mass. |
| `uniform` | Independent uniform density/mass draws from the configured intervals. Default: redraw each integration step. |
| `adversarial` | Choose parameter endpoints online for each controller and trajectory, based on the current state and commanded control. When both parameters are adversarial, search the four endpoint pairs jointly. |

Use `--uniform-refresh episode` to draw each uniform parameter once per
trajectory and hold it fixed. The default `--uniform-refresh step` matches
the pendulum runner's stepwise random disturbances. For example, fixed random
physical parameters in every rollout:

```bash
./.venv/bin/python -u examples/xv15_uncertain/run_mc.py \
    --density-modes uniform --mass-modes uniform --uniform-refresh episode \
    --mass-min 5841 --mass-max 5959
```

Two adversarial methods run by default: `lookahead` maximizes scaled distance
from the goal after one deterministic step, while `nearest_unsafe` minimizes
scaled distance to the unsafe union. Both prioritize immediate safety failure
and discourage goal entry. They use the same geometric objectives for every
controller, require no learned certificate, and do not observe future Brownian
increments. Nonadversarial coordinates retain their nominal value or sampled
uniform realization during the endpoint search. Select a single method with
`--adversarial-methods lookahead` or `--adversarial-methods nearest_unsafe`.
These are heuristic stress tests; endpoint selection does not establish the
worst possible reach-avoid probability over all parameter processes.

Initial states and Brownian increments are shared across controllers, training
seeds, modes, and attack methods for the same `--mc-seed` and `--n-mc`.
Separate random streams for initialization, density, mass, and Brownian noise
prevent changing a mode from changing the SDE noise. Draws retain trajectory
indices after other trajectories terminate. Brownian diffusion remains enabled
in `zero` mode; `--deterministic` disables diffusion independently of parameter
uncertainty.

### Metrics, cache, and plots

Trajectories stop on goal entry, unsafe entry, domain exit, nonfinite state,
or the finite horizon. Safety failure takes precedence over simultaneous goal
entry. Euler–Maruyama events are checked at step boundaries, including the final
partial step; crossings between integration steps can be missed. Check
sensitivity to `--dt` when interpreting results. MC estimates are empirical
finite-horizon results and do not replace the bound-training certificate.

The summary reports success, failure, and timeout rates. For scenarios with
adversarial coordinates, it selects the lowest success rate across tested
attacks **for each training seed**, then averages across seeds, as in the
pendulum runner. Failure/timeout and other summary metrics use those same
selected runs. Per-seed details retain both attacks. Hitting-time means include
successful trajectories only, and the reported mean averages the available
per-seed means. Control effort includes all outcomes, using
`integral [(T/(m0*g))^2 + (alpha/alpha_max)^2 + (delta/delta_max)^2] dt` up to
stopping. This normalized effort is not physical energy. Nonfinite effort
values are excluded from its mean and counted explicitly in `per_seed.csv`.

Outputs go to `run_mc_results/`, or `--output-dir PATH`:

- `mc_cache.pth`: individual outcomes, stopping times, effort, initial/final
  states, sampled paths, effective evaluation config, RNG seed, checkpoint
  paths, and attack settings.
- `summary.csv` / `summary.json`: comparison rows, selected attacks per seed,
  and (in JSON) evaluation metadata.
- `per_seed.csv`: all individual training-seed/scenario/attack statistics.
- `success_rate_summary.pdf`: density × mass success-rate heatmaps.
- `time_and_effort.pdf`: successful hitting times and normalized effort.
- `density_<mode>__mass_<mode>/`: `trajectories.pdf`, `controls.pdf`, and
  `parameters.pdf` for a limited set of cached paths; these traces are
  illustrative and can include different attack methods than a summary row.

`--n-paths` controls saved paths per training seed/scenario/attack (default 5),
and `--trace-dt` controls their recording cadence (default 0.1 s).
`--no-plots` saves the cache and tables only. Existing MC files with the same
names are overwritten; training checkpoints and output directories are not
cleared or modified. Use a distinct `--output-dir` to preserve multiple MC runs.

Regenerate tables and plots without rerunning simulations or loading controllers:

```bash
./.venv/bin/python examples/xv15_uncertain/postprocess_mc.py \
    --cache examples/xv15_uncertain/run_mc_results/mc_cache.pth
```

Optional postprocessing flags: `--output-dir PATH`, `--max-paths N` (plotted
paths per controller/scenario), and `--no-plots`.

Implementation checks:

```bash
MPLBACKEND=Agg ./.venv/bin/python -m pytest examples/xv15_uncertain/test_run_mc.py -q
```

## Run

From the repository root, using the shared Python environment:

```bash
./.venv/bin/python -u examples/xv15_uncertain/neural_certified_nominal_drift/main.py
```

The default seed is `TRAIN_SEED = 0` in `main.py`. It can also be selected
without editing the file:

```bash
./.venv/bin/python -u examples/xv15_uncertain/neural_certified_nominal_drift/main.py --seed 1
```

Defaults are 15,000 pretraining epochs with 1,200 samples per region category
and 30,000 bound-training epochs, with learning rates 0.01 and 0.005. The six
unsafe boxes share the unsafe sampling budget. Initial discretization budgets
are 1,000 cells per V region group and 1,000 generator cells; adaptive
refinement follows the nominal pendulum baseline's schedule. Training stops
at the first SAT reported by the shared bound trainer or the epoch limit.

Optional flags: `--pretrain-epochs N`, `--epochs N`, `--device cuda`,
`--config PATH`, and `--no-plots` (also disables progress plots). For example:

```bash
./.venv/bin/python -u examples/xv15_uncertain/neural_certified_nominal_drift/main.py \
    --seed 0 --pretrain-epochs 15000 --epochs 30000 --no-plots
```

Each run starts fresh and clears that seed's `outputs/`, `results/`, and
`training_progress/` directories, matching the pendulum baseline. Use a new
seed or copy the existing run directory before rerunning an experiment you
want to keep.

## Problem and neural models

`config.json` contains the nominal physical constants, aerodynamic
coefficients, diffusion, control limits, regions, and `beta_ra = 5` from
`examples/synthesis/xv15aircraft_syn/main.py`.

| Region | Airspeed (m/s) | Flight-path angle (deg) | Rotor tilt (deg) |
|---|---|---|---|
| Domain | [0.5, 100] | [-20, 20] | [0, 90] |
| Initial | [28, 32] | [8.5, 10.5] | [58, 62] |
| Goal | [65, 85] | [-2, 10] | [25, 35] |

Six unsafe strips cover the domain boundary: thickness 0.5 m/s at each
airspeed face and 1 degree at each angular face.

Configuration angles are written in degrees for readability. Runtime states
are `[v, gamma, beta]` in `[m/s, rad, rad]`. Control outputs are
`[T, alpha, delta]` in `[N, rad, rad/s]`. The independent diffusion amplitudes
are `[0.5, 0.1*pi/180, 0.1*pi/180]` in runtime units per square root second.
The drift evaluates the full aircraft equations `f(x, u(x))`, including
lift/drag and division by positive airspeed. No velocity clamp or additive
control approximation is used.

### 1. The certificate model

The certificate is the value network `V` (`src/network.py`'s `V_offset`,
built by `create_V`): a sigmoid MLP, 3 inputs → 64 → 64 → 1 output, with
input scales `[100, 20*pi/180, 90*pi/180]` (so raw `[v, gamma, beta]` errors
from trim are normalized to roughly unit range) and output scale 20 (hidden
activations are multiplied by 20 before the output layer, giving the sigmoid
units room to saturate and `V` room to vary steeply). `V_offset` is
constructed so that it is exactly `0.1` at the nominal trim state
`x_eq = [75 m/s, 4 deg, 30 deg]` by subtracting the network's own output at
`x_norm = 0` and adding back `output_offset = 0.1`; this holds by
construction (verified numerically by `verify_zero_at_offset`), not by
training.

Paired with `V` is the generator network `GV` (`src/phi_module.py`'s
`GV_offset`, built by `create_GV`), which is not a second set of learned
weights: it analytically evaluates the infinitesimal generator of `V` under
the closed-loop SDE,
`(GV)(x) = f_cl(x)·∇V(x) + 1/2 * Tr(g(x) g(x)^T H_V(x))`,
by propagating `V`'s sigmoid Jacobian/Hessian in closed form (no autograd
graph at eval time). `check_generator` in `main.py` cross-checks this
analytical `GV` against PyTorch autograd on random in-domain points, domain
corners, and the trim point before training starts, so a broken generator
implementation would be caught immediately rather than silently poisoning
the certificate loss.

Together `(V, GV)` play the role of a reach-avoid stochastic barrier
(supermartingale) certificate: `V` assigns a scalar "risk level" to each
state, and `GV` measures whether that risk is expected to decrease along
trajectories. `beta_ra = 5` (from `config.json`) is the threshold separating
"safe/still trying to reach goal" (`V < beta_ra`) from the certified-unsafe
level set (`V >= beta_ra`); see Q3 below for the exact conditions `V`/`GV`
must satisfy.

### 2. The controller model

The controller is `XV15EqMLPControl` (`model.py`), a bias-free 3→64→3 tanh
MLP (`fc1`, `fc2` have no bias terms, so `u(x_eq) = u_eq` exactly regardless
of the trained hidden weights). Its forward pass:

1. Normalizes the state error `(x - x_eq) / input_scale` (state scales, the
   same `params.network.input_scale` used by `V`).
2. Passes it through `fc1 → tanh → fc2` to get a raw correction `du`, scaled
   by `du_scale = [T_max, alpha_max, delta_max]`.
3. Adds `du` (after dividing by the trim-point slope of each output
   nonlinearity, `inv_slope_eq`) to `z_eq`, the pre-nonlinearity trim value,
   then applies per-channel sigmoid (thrust) / tanh (angle of attack, tilt
   rate) squashing so outputs never leave their physical bounds:
   thrust ∈ `[0.1, 1.8] * mass * gravity` N, angle of attack ∈ `±16°`, tilt
   rate ∈ `±5°/s`.

The `inv_slope_eq` scaling is what makes `du` act like a *linear* correction
near trim (a unit change in `du` produces approximately a unit change in the
physical output there), while the sigmoid/tanh saturate the output only as
the state moves away from trim — this is why the docstring calls it
"equilibrium-centered, slope-matched." `x_eq`/`u_eq` come from
`find_goal_equilibrium`, which solves the nominal trim equations
(`brentq` on the pitch/lift/drag/weight residual) at the center of the goal
region, `[75 m/s, 4 deg, 30 deg]`, not from any learned or fitted data.
Controller weights (`fc1`, `fc2`) are otherwise randomly initialized and
trained jointly with `V` by the same bound-training loop.

Two initialization/normalization choices differ from the original aircraft
script (`examples/synthesis/xv15aircraft_syn/main.py`): trim is solved at
the goal center rather than elsewhere, and controller inputs are normalized
by state scales rather than control limits. These choices leave the SDE and
reach-avoid regions unchanged.

### 3. How training guarantees the certificate conditions

`train_network_bounds` (`src/trainer.py`) does not rely on samples to decide
success. Each epoch it computes CROWN/IBP bounds (`src/crown_bounds.py`,
currently auto_LiRPA IBP) on `V` and `GV` over every cell of an exhaustive
discretization of the state-space regions (`discretize_regions`, refined
adaptively when cells fail — see `params.refinement` in `main.py`). Because
IBP bounds hold for *all* points inside a cell's box, not just its sampled
corners, a bound that satisfies a condition over every cell in a region
proves the condition for every real-valued state in that region, not merely
at the points evaluated.

`compute_total_loss_bounds` (`src/training_utils.py`) turns those bounds
into five conditions, each with its own `sat_*` flag computed straight from
the bounds (`>= `/`<=`, not from a loss threshold):

- **Goal**: `V_lower(x) >= 0` for all `x` in the goal region.
- **Init**: `V_upper(x) <= 1` for all `x` in the init region.
- **Unsafe**: `V_lower(x) >= beta_ra` for all `x` in the unsafe region.
- **Outside** (domain minus goal): `V_lower(x) >= 0`.
- **Generator**: `GV_upper(x) < 0` for every generator cell whose
  `V_lower(x) < beta_ra` (cells already certified `V >= beta_ra` are exempt,
  since the supermartingale condition is only needed while a trajectory is
  still "at risk").

`_compute_all_satisfied` (`src/trainer.py`) ANDs all five `sat_*` flags
together each epoch. With `curriculum_mode = "none"` (this baseline), the
loop `break`s and saves a checkpoint the *first* epoch every flag is `True`
simultaneously — training does not keep going past that point, and it does
not run a fixed number of epochs and then check at the end. If the epoch
budget (30,000) is exhausted first, no SAT is claimed; `eval_bundle.pth` and
the terminal log still record the final (possibly failing) bounds.

Because these are the same soundly-propagated bounds used for the loss
(rather than a separate looser check), a run that terminates with "first
SAT" is, by construction, a formal proof that `(V, GV)` satisfy all five
inequalities everywhere in their respective regions under the trained
`controller` and the nominal closed-loop SDE — which is exactly the
reach-avoid supermartingale certificate condition. It is *not* a promise
that training will ever reach SAT, and it says nothing about states outside
the fixed nominal dynamics (see the parameter-uncertainty caveat below).

Two things this does **not** by itself establish: sample checks/plots (used
during pretraining and for visualization) are diagnostic only, not part of
the SAT decision; and a nominal-parameter SAT does not certify robustness to
parameter uncertainty — that is the subject of later experiments in this
directory.

## Animation

`animate.py` loads a trained seed's `outputs/eval_bundle.pth` (controller
and certificate weights) and `outputs/run_config.json` (the exact config/
hyperparameters that run used), rebuilds `XV15Aero`, `XV15EqMLPControl`, the
certificate `V`, and the analytical generator `GV` from `model.py` and the
shared `src/` modules (`create_V`, `create_GV`, cross-checked once against
autograd via `verify_GV` before simulating), then runs one closed-loop
Euler-Maruyama rollout from the init region. It renders the rollout into a
single self-contained `animate.html` (own HTML/canvas/JS, no server and no
external assets — open the file directly in a browser), in the same spirit
as `nova_3d_xv15_syn`'s rollout animation but implemented independently
against this example's own model: a side-view flight-path canvas with an
aircraft glyph (pitch = gamma+alpha, rotor mark = beta, flame length =
thrust), a stacked time-series canvas (airspeed, gamma, beta, each shaded
with the projected initial/goal/unsafe intervals), and a certificate strip
plotting `V(x(t))` against the `V=0`/`V=1`/`V=beta_ra` reference lines from
Q3 above. Play/pause, restart, a scrub bar, and a speed selector control
playback; a badge in the header reports whether *that seed's weights* passed
the bound-checked training SAT (from `eval_bundle.pth`'s `final_results`),
which is a separate, stronger claim than anything about this one sampled
rollout.

```bash
./.venv/bin/python -u examples/xv15_uncertain/animate.py --seed 0
```

Output goes to `neural_certified_nominal_drift/seed0/results/animate.html`.
Useful flags: `--seconds` (simulated horizon), `--dt` (integration step),
`--frame-dt` (recorded-sample spacing), `--sim-seed` (rollout RNG,
independent of `--seed`), `--deterministic` (disable SDE noise), and
`--out-name` to write under a different file name, e.g.:

```bash
./.venv/bin/python -u examples/xv15_uncertain/animate.py \
    --seed 0 --seconds 15 --sim-seed 1 --out-name rollout_1.html
```

This is a simulation for illustration, not a verification proof: it uses a
fixed Euler-Maruyama step and one sampled initial condition, unlike the
exhaustive bound propagation described in Q3. A run's outcome (reaching
goal, hitting an unsafe box, exiting the domain, or timing out) does not
change the bound-checked SAT badge, which is fixed once training saved that
seed's `eval_bundle.pth`.

## Outputs and checks

For seed 0, outputs are under `neural_certified_nominal_drift/seed0/`:

- `outputs/terminal_log.txt`: setup, generator check, pretraining, training,
  final constraint summary, and errors.
- `outputs/run_config.json`: nominal problem and training settings.
- `outputs/V_pretrained.pth`, `outputs/controller_pretrained.pth`: sample-stage weights.
- `outputs/eval_bundle.pth`: final weights, region cells, hyperparameters,
  training history, and evaluation, saved even if the epoch budget ends without SAT.
- `outputs/resume_checkpoint.pth`: written by the shared trainer at SAT;
  this baseline does not load it automatically.
- `training_progress/` and `results/`: progress and final plots unless disabled.

Implementation checks (no training):

```bash
MPLBACKEND=Agg ./.venv/bin/python -m pytest examples/xv15_uncertain/test_nominal_drift.py -q
```

They compare the drift with the original aircraft implementation, check
units and control limits, compare the generator with autograd, and exercise
bound enclosures and gradients through both neural networks.
