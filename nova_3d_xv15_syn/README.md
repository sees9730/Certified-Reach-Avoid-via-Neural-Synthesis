# Smooth XV-15 certificate and controller synthesis

Self-contained code: nothing is imported from `src/`, or the original
example. 
The controller is **continuous and C-infinity on the full domain**
(airspeed is strictly positive there). 
Certificate and controller parameters are trained together using adaptive cells and bound losses.

`train` initializes a new certificate and controller
and recreates `results/`. 
Run training before using `verify`, `plot`, or either animation command, since those commands need a saved candidate.

V is globally nonnegative by construction; the independent verifier checks the
certificate inequalities and control limits for each new run. 
Current commands require `controller_type = smooth_affine_softmin` in the saved artifact.

## Run

From the repository root with the existing environment:

```bash
./.venv/bin/python -m nova_3d_xv15_syn train
./.venv/bin/python -m nova_3d_xv15_syn verify
./.venv/bin/python -m nova_3d_xv15_syn animate
./.venv/bin/python -m nova_3d_xv15_syn plot
./.venv/bin/python -m unittest nova_3d_xv15_syn.test_nova_syn -v
```

`train` synthesizes, independently verifies, plots, and animates. 
`verify` reloads the saved pair and generates a fresh report, plots, and animation. `animate` redraws the rollout without rerunning the proof. 
Open **`nova_3d_xv15_syn/results/aircraft_animation.html`** in a browser for playback, scrubbing, and speed controls; it works offline without external scripts or a server.

```bash
# Another stochastic rollout; proof still covers the original SDE.
./.venv/bin/python -m nova_3d_xv15_syn verify --animation-seed 4
# Longer horizon or finer integration.
./.venv/bin/python -m nova_3d_xv15_syn animate --animation-time 90 --animation-dt 0.0005
# Drift-only visualization, explicitly labeled in the animation.
./.venv/bin/python -m nova_3d_xv15_syn animate --deterministic
# Verify and plot without a rollout.
./.venv/bin/python -m nova_3d_xv15_syn verify --no-animation
```

Dependencies are managed by the repository's shared
[root requirements.txt](../requirements.txt). Install from the repository root;
direct script execution also works:

```bash
python -m pip install -r requirements.txt
python nova_3d_xv15_syn/main.py train --output nova_3d_xv15_syn/results
```

The source folder remains self-contained. To run a copy elsewhere, first install
the dependencies from the root requirements file in that environment.

The proof-only entry point uses just `mpmath` and the standard library:
`python nova_3d_xv15_syn/verify.py nova_3d_xv15_syn/results/candidate.json`.

Search budgets: `--steps`, `--seconds`, `--max-cells`, `--refine-every`, and
`--refine-batch`. An incomplete proof returns **UNKNOWN**, not infeasibility.
This finite model family has no general convergence guarantee.

## Exact benchmark and units

`problem.json` reproduces the XV-15 task in
`examples/synthesis/xv15aircraft_syn/main.py`, with **beta_ra = 5**.
Angles are represented internally in **degrees**; angular drift and diffusion
are converted consistently from the source radian coordinates.

| Region | Airspeed [m/s] | Flight-path angle [deg] | Rotor tilt [deg] |
|---|---|---|---|
| Domain | [0.5, 100] | [−20, 20] | [0, 90] |
| Initial | [28, 32] | [8.5, 10.5] | [58, 62] |
| Goal | [65, 85] | [−2, 10] | [25, 35] |

The unsafe set is the union of six full-width strips: airspeed in [0.5, 1]
or [99.5, 100], flight-path angle in [−20, −19] or [19, 20], and rotor tilt
in [0, 1] or [89, 90]. Each spans the full domain in the remaining coordinates.

Independent constant noise amplitudes are **[0.5, 0.1, 0.1]** in these units.
Mass is 5900 kg, gravity 9.81 m/s², wing area 15.7 m², and density 1.225 kg/m³.
The full aerodynamic equations appear in `model.drift` and independently in
`verify.Checker.evaluate`; there is no velocity clamp.

Controls are normalized thrust `T/(mass*gravity)`, angle of attack in degrees,
and rotor-tilt rate in degrees/second. Their limits are **[0.1, 1.8]**, **[−16, 16]**,
and **[−5, 5]**; physical thrust ranges from **5787.9 N to 104182.2 N**.
Stored endpoints are rounded inward to respect exact decimal limits.

## Certificate function model

`V` has **18 learned nonnegative coefficients** (`weights`, shape `3x6`):

```text
z_i = (x_i - center_i) / scale_i
V(x) = sum_(i,j) weights[i,j] * (exp(rates[j]*z_i) - 1)^2 / normalizers[i,j]
```

`center` (goal center), `scale`, `rates`, `normalizers` are fixed, not
learned. Every term is squared and `weights >= 0` is reclamped after every
optimizer step, so **V ≥ 0 on all of R³ by construction** — training never
has to learn this, and verification never has to check it.

## Controller model

Runtime evaluation is fully analytic — no argmin, action bank, or clipping.

Angle of attack has its own **10 learned coefficients** (`alpha_weights`):

```text
phi(z) = [1, z1, z2, z3, z1², z2², z3², z1*z2, z1*z3, z2*z3]
alpha(x) = midpoint_alpha + half_range_alpha * tanh(alpha_weights . phi(z))
```

Thrust and tilt rate aren't separately learned; for this `alpha` the drift is
affine in both, so they're read off `V`'s gradient and soft-bounded:

```text
c_T = gravity * (V_v*cos((alpha+tilt)*pi/180)
                 + V_gamma*sin((alpha+tilt)*pi/180)/(v*pi/180))
c_delta = V_tilt
u_i(x) = midpoint_i - half_range_i * tanh(half_range_i*c_i(x)/eta)
```

Each channel is pushed toward whichever endpoint locally decreases `GV`,
smoothly. `eta = 0.001` is a fixed constant (not trained); smaller `eta`
sharpens the transition, with no bound on control derivatives.

## Training (how training reaches a SAT-valid certificate)

1. **Initialize**: an LP picks `weights` separating initial from unsafe by
   value alone; `alpha_weights` is least-squares fit to pointwise-searched
   low-`GV` proposals.
2. **Point/warmup training** (`--warmup`, default 250 steps): gradient steps
   on random points, same losses as step 3 but pointwise.
3. **Whole-cell bound training**: the domain is covered by a `Partition`
   (axis-aligned cells, pre-split at goal/unsafe boundaries). Each step,
   `Model.bounds` computes interval enclosures of `V` and of `GV` at the four
   thrust/tilt-rate corners (same shared smooth `alpha(x)`), takes the `min`
   corner, and adds the soft-min correction `2*eta*log(2)`. The loss
   penalizes `V` too high on initial / too low on unsafe, and — on active,
   not-yet-safe cells — corrected `GV` exceeding `-epsilon`. `weights >= 0`
   is reclamped after every step.
4. **Refine** (every `--refine-every` steps): among violating cells, only
   splits ones where the bound is genuinely loose (not just optimizer
   slack), picks the worst `--refine-batch`, and the axis with the best
   worst-child bound, growing the partition up to `--max-cells`.
5. **Independent verification** (`verify.py`, run automatically after
   `train`) recomputes the same inequalities from the saved artifact alone,
   replays the recorded partition splits, and is what actually reports SAT.

**Termination.** Step 3 stops on whichever comes first:
- **Self-check pass**, tested every `--refine-every` steps: `V_initial_upper
  <= 1`, `V_unsafe_lower >= beta_ra`, and no active cell's corrected `GV`
  exceeds `-epsilon`.
- `--steps` reached (default 10000), or `--seconds` wall-clock elapsed
  (default 600).

The self-check is *not* automatically SAT, even though it checks the same
inequalities as `verify.py`: it's computed with ordinary float64 torch
arithmetic (`interval.py`, "differentiable float64 interval proposals") using
float64-truncated physics constants, not `verify.py`'s 50-digit
outward-rounded intervals with exact decimal constants. In practice the loss
targets a small margin inside the true thresholds (e.g. `V<=0.95` on initial,
not `1`), which is normally enough slack to absorb that precision gap — but
`verify.py` is what's actually trusted, so it always reruns independently
regardless of why or when training stopped.

## Animation and saved outputs

The default rollout starts at `[30, 9.5, 60]`, the initial-box center, uses the
original SDE noise with seed 0, and integrates with Euler-Maruyama at **0.001 s**.
It stops at the first detected goal/unsafe/domain-exit event or at 60 seconds.
Events are checked at integration steps. This numerical illustration does not
establish or change SAT. Select a smaller step to inspect sensitivity to rapid
control variation. The supplied rollout reaches the goal at approximately 7.40 s.

The aircraft schematic shows body pitch `gamma + alpha`, rotor tilt relative to
the body, and thrust arrows. Display position integrates `v*cos(gamma)` and
`v*sin(gamma)` starting from zero relative altitude. These display coordinates
are not extra certified states. State traces show initial, goal, and unsafe
ranges. Controls are displayed directly without visual smoothing.

- `results/candidate.json`: smooth controller, certificate, problem, partition,
  proof witnesses, and parameter-change measurements.
- `results/verification.json`: SAT/UNKNOWN, bounds, per-cell results.
- `results/training.csv`: optimization and refinement history.
- `results/slices.pdf`, `results/slice_*.png`: V/GV contours, policy slices, regions.
- `results/aircraft_animation.html`: self-contained interactive animation.
- `results/rollout.json`, `results/rollout.csv`: displayed trajectory samples,
  about 0.05 s apart, and reproducibility metadata.

Load the controller directly:

```python
import json
import torch
from nova_3d_xv15_syn.model import Model

saved = json.load(open("nova_3d_xv15_syn/results/candidate.json"))
model = Model(saved["problem"], initialize=False)
model.restore(saved["model"])
with torch.no_grad():
    thrust_ratio, alpha_deg, tilt_rate_deg_s = model.policy([30.0, 9.5, 60.0])
    thrust_newtons = thrust_ratio.lo.item() * 5900 * 9.81
    V, GV = model.values([30.0, 9.5, 60.0])
```

Changing regions/thresholds requires a fresh synthesis run. Changing the SDE
requires updating both dynamics implementations and their consistency test.
The smooth bound specifically uses affine dependence on thrust and tilt rate;
other dynamics require an appropriate bound.
