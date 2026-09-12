# Certificate and smooth-controller synthesis for a stochastic double integrator

This example synthesizes a scalar certificate **V(px, py, vx, vy)** and a smooth
state-feedback controller **u(px, py, vx, vy) = [ux, uy]**. The certificate is a
polynomial in two transformed energy coordinates. The controller is an analytic
feedback law that bends motion around an obstacle and applies smooth saturation.
The synthesis procedure combines a finite search over controller parameters,
linear programming for certificate coefficients, and adaptive interval bounds.

The code is self-contained in this folder. It does not import `src/`, `nova_5d_gbm_veri/`, or
`nova_3d_xv15_syn/`. The current implementation uses these structured models; it does not
train a neural network or use gradient descent.

## Verified result

The reported run achieved **SAT**. Its report fingerprint matches the saved
[results/candidate.json](results/candidate.json):

```text
8bbee31b4a485cdb079f36cd617e9e40544a6a206474c2ce0e5620de3b679707
```

| Condition | Verified bound | Required |
|---|---:|---:|
| V on the complete initial region | upper ≤ 0.9714842772724744 | ≤ 1 |
| V on the complete unsafe region | lower ≥ 5.1783051339896495 | ≥ 5 |
| GV on active cells outside the goal | upper ≤ −0.00010100000000000025 | ≤ −0.0001 |
| Deployed control, each component | absolute value ≤ 20 | ≤ 20 |

The checker also proved global nonnegativity of V, containment of `{V < 5}` in
the full domain, and the validity of the raw-controller generator identity on
that sublevel set. There were **248 energy cells**, **141 initial/unsafe cells**,
and **158 accepted control-cover cells**, with no failed conditions. The raw
control bounds on the sublevel set were approximately `[3.98770, 18.99460]`,
inside the saturation's identity core of `[-19, 19]`.

The resulting reported reach-before-unsafe probability lower bound is **0.8**,
using `V_initial <= 1` and `beta_ra = 5`. This is a bound for the continuous-time
SDE, with no prescribed completion time. It is not an empirical rollout success
rate. The saved training log ends at step 23 after approximately 5.78 seconds;
that timer includes proposal search and refinement, but excludes final independent
verification, plotting, and Python startup. These figures describe this saved run.

## 1. Reach-avoid problem

The state supplied to both models is a four-vector in this exact order:

```text
x = [px, py, vx, vy]
```

`px, py` are positions; `vx, vy` are velocities. The controller outputs two
accelerations. Two independent Brownian motions perturb the accelerations:

```text
dpx = vx dt                       dpy = vy dt
dvx = ux dt + 0.20 dWx             dvy = uy dt + 0.25 dWy
```

The full specification is in [problem.json](problem.json).

| Region | px | py | vx | vy |
|---|---|---|---|---|
| Domain | [-5, 5] | [-5, 5] | [-8, 8] | [-8, 8] |
| Initial | [-3.1, -2.9] | [1.4, 1.6] | [0.3, 0.5] | [-1.17, -0.97] |
| Goal | [-0.5, 0.5] | [-0.5, 0.5] | [-0.5, 0.5] | [-0.5, 0.5] |
| Interior unsafe obstacle | [-1.9, -1.1] | [0.65, 1.45] | [-8, 8] | [-8, 8] |

The unsafe set also contains eight boundary strips of thickness 0.25: position
coordinates in `[-5, -4.75]` or `[4.75, 5]`, and velocity coordinates in
`[-8, -7.75]` or `[7.75, 8]`, spanning the other coordinates' full ranges.
They cover every face, edge, and corner of the domain.

The initial region and goal are disjoint. The straight position segment from
`(-3, 1.5)` to `(0, 0)` intersects the interior obstacle. Reaching the goal requires
all four coordinates to enter their goal intervals simultaneously.

The certificate requirements are:

1. `V(x) >= 0` for every state.
2. `V(x) <= 1` throughout the initial region.
3. `V(x) >= beta_ra = 5` throughout every unsafe box.
4. `GV(x) <= -epsilon = -0.0001` wherever `V(x) < 5` outside the goal.

Here **GV is the infinitesimal generator of V under the deployed controller**:

```text
GV = V_px*vx + V_py*vy + V_vx*ux + V_vy*uy
     + 0.5*(0.20^2*V_vx_vx + 0.25^2*V_vy_vy)
```

The diffusion term is part of both training bounds and verification.

## 2. What is the certificate model V?

### Curved coordinates

[CurvedPair in curved.py](curved.py) constructs a cubic reference curve:

```text
anchor = initial position center's px
slope  = initial position center's py / anchor
h(px)  = (slope + A)*px - A*px^3/anchor^2
```

For every amplitude `A`, this curve passes through the initial position center
and the origin. `A` changes its bend. It is a geometric reference curve, not a
trajectory indexed by time.

Define the following quantities from the full input state:

```text
s  = py - h(px)                   # position error from the curve
v_perp = vy - h'(px)*vx           # drift rate of that position error
wx = vx + kx*px
wy = v_perp + ky*s

qx = px^2 + gamma_x*wx^2
qy = s^2  + gamma_y*wy^2
```

`qx` measures longitudinal position/velocity energy. `qy` measures deviation
from the curve and the corresponding transverse velocity. Both are nonnegative.
Their curved level sets allow the certificate to exclude the blocking obstacle.

### Polynomial certificate

The model has **20 nonnegative coefficients**, ten for each energy:

```text
Fx(qx) = sum(j=1..10) ax[j] * (qx/Rx)^j
Fy(qy) = sum(j=1..10) ay[j] * (qy/Ry)^j
V(x)   = Fx(qx(x)) + Fy(qy(x))
```

`Rx, Ry` are positive numerical scales computed from initial-region energy upper
bounds when the model is initialized. They remain fixed during the controller
parameter search. They are distinct from the energy-cover caps used by verification.

There is no constant term. Nonnegative coefficients and nonnegative energies
make **V globally nonnegative by construction**, and `V(0) = 0`. The basis has
degree 10 in each energy; because the coordinate transformation is nonlinear,
this is not a degree-10 polynomial in the original state coordinates.

### The actual learned V in the SAT run

The saved model has `kx = 0.2`, `ky = 1`, `gamma_x = 1.3`, `gamma_y = 1`, and
`A = 1.1`. Thus its coordinates are

```text
h(px) = 0.6*px - (1.1/9)*px^3
s     = py - h(px)
v_perp = vy - (0.6 - (1.1/3)*px^2)*vx
qx    = px^2 + 1.3*(vx + 0.2*px)^2
qy    = s^2 + (v_perp + s)^2

rx = qx / 9.743120000000001
ry = qy / 0.853094128024693
```

Only six of the fitted coefficients are nonzero:

```text
V(x) = 0.07811010257730344    * rx
     + 0.7398173662622093     * rx^10
     + 0.0017254388053788944  * ry
     + 0.047136370145954794   * ry^2
     + 2.7383880925697928     * ry^9
     + 0.2994156619115181     * ry^10
```

These numbers come from the fingerprinted candidate above. In its JSON model,
`weights[0][j-1]` is `ax[j]` and `weights[1][j-1]` is `ay[j]`. The saved decimal
strings define exact rational parameters for the checker. Load the saved model
for deployment rather than transcribing rounded equations.

## 3. What is the controller model? What are its inputs?

The controller is a **memoryless, time-independent, smooth nonlinear state-feedback
function from R⁴ to R²**:

```text
input:  [px, py, vx, vy]
output: [ux, uy]
```

It computes `h`, `s`, and `v_perp` from the state. It does not require a time index,
a Brownian increment, a rollout history, or V as an additional input. The problem
geometry affects the synthesized parameters; regions are not additional runtime
inputs. Once synthesis is finished, evaluating the controller requires no optimizer.

### Raw feedback law

Before saturation, the accelerations are

```text
ux_raw = -(1/gamma_x + kx*dx)*px - (kx + dx)*vx

uy_raw = h'(px)*ux_raw + h''(px)*vx^2
         - (1/gamma_y + ky*dy)*s - (ky + dy)*v_perp
```

The first law stabilizes longitudinal motion. The terms involving `h'` and `h''`
compensate for the bending coordinates; the remaining terms stabilize transverse
position and velocity. The raw feedback is polynomial in the state.

For the saved SAT model, `dx = 0.5` and `dy = 6`, so

```text
ux_raw = -(1/1.3 + 0.1)*px - 0.7*vx
uy_raw = (0.6 - (1.1/3)*px^2)*ux_raw
         - (2.2/3)*px*vx^2 - 7*s - 7*v_perp
```

### Smooth bounded output

The deployed output is `ui = sat(ui_raw)` independently for each component.
For a component limit `L`, let `core = 0.95*L` and `gap = L-core`:

```text
sat(a) = a                                           if abs(a) <= core
sat(a) = sign(a) * (core + gap/(gap/r + exp(-gap/r)))   otherwise
         where r = abs(a)-core > 0
```

This saturation is C-infinity, equals the identity in its core, and bounds the
magnitude by `L` globally. For `L = 20`, the identity core is `[-19, 19]`.
The difference from the identity is flat at the core boundary, so that boundary
does not introduce a kink. The checker proves that raw feedback stays in this
core on `{V < 5}`; consequently the generator identities used in the proof
apply to the actual deployed controller there. `runtime.py` includes the
saturation correction when evaluating GV elsewhere.

## 4. How are V and the controller trained?

### Which parameters change?

| Quantity | Role | Treatment in the current implementation |
|---|---|---|
| `A` | Curve amplitude; affects both V and controller | Search `curve_amplitudes` in `search.json` |
| `dy` | Transverse damping in the controller | Search `transverse_damping` in `search.json` |
| `ax[1..10], ay[1..10]` | Certificate coefficients | Fit by linear programming, then optionally rescale using bounds |
| `dx` | Longitudinal damping | Set by `longitudinal_damping` in `search.json` |
| `kx, ky` | Position gains in coordinates and feedback | Fixed at `[0.2, 1]` in `curved.py` |
| `gamma_x, gamma_y` | Energy weights | Fixed at `[1.3, 1]` in `curved.py` |
| `anchor, slope` | Curve endpoints | Computed from the initial position center |
| `Rx, Ry` | Polynomial normalization | Computed once at model initialization |

Thus the pair is synthesized by **controller parameter search with a certificate
fit for each proposed controller**. The code does not continuously optimize every
controller gain. In the successful run, the ninth parameter pair, `(A, dy) =
(1.1, 6)`, was accepted. Its amplitude equals the constructor's initial amplitude,
so `parameter_changes.amplitude = 0` even though alternatives were searched;
`controller_search` records those attempts.

### Stage A: propose a controller and fit V

[bound_training.py](bound_training.py) enumerates the Cartesian product of the
configured amplitude and transverse-damping lists in their listed order. For a
fixed controller, V and its generator bounds are linear in the 20 coefficients.
It solves the linear program

```text
minimize    sum of all certificate coefficients
subject to  coefficients >= 0
            finite initial-corner V constraints
            whole-box V lower bounds for selected interior obstacles
            generator upper bounds at selected energy levels outside the goal enclosure
            Fx(cap_x) and Fy(cap_y) large enough to cover the sublevel set
```

The default proposal targets are `V <= 0.95` at initial corners, `V >= 5.2` for
those interior-obstacle bounds, `GV_upper <= -0.00012`, and
`Fi(cap_i) >= 5.02`. The energy samples include a 150-by-150 nonuniform grid and
additional points near the boundary of the conservative goal enclosure.
A bound at an energy level covers the states represented by that level, but
these finitely many levels alone do not cover the continuous state space.

The independent interval routine then screens the proposal for sublevel domain
containment and raw-control core containment. The first proposal passing both
fitting and this screen proceeds to refinement. No previous result is loaded.

### Stage B: evaluate complete covers and refine failing cells

The algorithm starts with all ten region boxes (one initial and nine unsafe)
and the entire normalized energy square `[0,1] × [0,1]`. The energy square maps
to `qx ∈ [0,11.8]`, `qy ∈ [0,1.2]` with the default caps. These are energy
coordinates, not a replacement physical domain.

On every iteration it computes initial V upper bounds, unsafe V lower bounds,
and generator upper bounds. An energy cell needs a generator check when its
`V_lower < beta_ra` and its entire represented state set has not been proved
inside the goal. In particular, a lower bound equal to beta excludes that cell;
a generator upper bound of zero fails the required negative margin.

Failing cells are prioritized by violation. Each selected cell is bisected along
its longest normalized eligible side. Energy cells split in either energy
coordinate. Region cells split in `px` or `vx`; analytic quadratic extrema still
bound every `py` and `vy` value in the cell. Their intervals are never discarded.
The partition is retained, with every split recorded; this implementation does
not merge cells.

During this phase, the controller and relative polynomial weights stay fixed.
If the worst generator upper bound `g` is already negative, the algorithm may
scale **all** certificate weights by a positive factor

```text
c = max(1, 1.01*epsilon/(-g), 1.01*beta_ra/V_unsafe_lower)
```

It applies this update only when the unsafe lower bound is positive, `c > 1`,
and the scaled initial upper bound is at most `0.995`. Scaling multiplies V and
GV by the same factor and shrinks the active sublevel set. Extra region
refinement targets (`0.955` for initial and `1.01*beta_ra` for unsafe) help expose
room for this update. This is the bound-based certificate update in refinement;
the full linear program is not solved again on every refinement iteration.

The logged `loss` is a diagnostic sum of positive violations of the three bound
thresholds. It is not an objective differentiated by an optimizer in this phase.
A genuinely positive generator bound that persists as cells shrink may require
another model or proposal; more subdivision alone cannot fix a bad controller.

### Stage C: independent verification

When floating-point bounds satisfy the thresholds, or a search budget expires,
the candidate and all partitions are saved. `check.py` dispatches to
[curved_check.py](curved_check.py), which uses exact rational interval arithmetic
and rational square-root enclosures to recompute the proof.

For every energy cell it proves one of:

1. The entire represented state set lies in the goal.
2. `V_lower >= beta_ra`.
3. `GV_upper <= -epsilon`.

Nonnegative polynomial coefficients and `Fi(cap_i) >= beta_ra` ensure the
energy rectangle covers the entire strict sublevel set. A separate interval
cover in `(px, wx)` uses the remaining V budget to bound all transverse states
and prove physical domain containment and raw-control core containment. The
initial and every unsafe box are checked in full. Replaying splits from their
root boxes ensures that verification cannot omit gaps in the cover.

Only this checker returns **SAT**. The floating-point stop message
`independent exact verification required` describes the handoff to the checker;
it is consistent with the final report subsequently returning SAT. A null
`GV_upper` in a report cell means that the goal or V-threshold condition excluded
that cell from the generator check, not that its generator is zero.

### Termination

`learn` (in `bound_training.py`) stops for exactly one of four recorded
`training_stop` reasons:

1. **`"Floating-point bounds satisfied; independent exact verification
   required"`** — Stage B's self-check passes (`initial<=1`, `unsafe>=beta_ra`,
   `worst<=-epsilon`). This is the normal "try Stage C now" signal, not SAT
   itself: it's computed with ordinary `torch.float64` interval arithmetic
   (`curved.py`, `B`), not `curved_check.py`'s exact `Fraction` arithmetic. The
   `0.955`/`1.01*beta_ra` refinement targets and generator margin factor are
   built-in slack meant to absorb that gap, but Stage C is what's actually
   trusted, and it always reruns regardless of why Stage B stopped.
2. **`"No feasible proposal within the controller search and budget"`** — no
   `(amplitude, damping)` combination in `search.json` both fit an LP and
   passed the control-cover screen before the time budget ran out.
3. **`"Cell budget exhausted"`** — every energy and region cell group hit
   `max_cells` while cells still failed their bound.
4. **`"budget exhausted"`** (default) — `--steps` or `--seconds` ran out in
   Stage B without either of the above.

A candidate is always saved and always passed to Stage C, whichever reason
applies — including reason 2, where verification runs against a fallback
complete cover (`model.caps(beta_ra)`) and is expected to report `UNKNOWN`.

## 5. Run training, verification, and visualization

From the repository root, with the dependencies installed:

```bash
# Synthesize and independently verify; omit visualization for this run.
./.venv/bin/python -m nova_4d_double_integrator_syn train --no-plots

# Independently verify the saved pair again.
./.venv/bin/python -m nova_4d_double_integrator_syn verify --no-plots

# Generate V/GV slice plots, rollout plots, and HTML animation.
./.venv/bin/python -m nova_4d_double_integrator_syn plot

# Regenerate only the rollout data and HTML animation.
./.venv/bin/python -m nova_4d_double_integrator_syn animate
```

`train` and `verify` also generate plots and animations unless `--no-plots` is
specified. `animate` uses saved verification metadata when its fingerprint
matches; it does not rerun the proof. The HTML works offline with no server.

The current CLI defaults in [main.py](main.py) are `--steps 1000000` and
`--seconds 1800`. An explicit shorter experiment that preserves the successful
run in `results/` is:

```bash
./.venv/bin/python -m nova_4d_double_integrator_syn train \
    --steps 4000 --seconds 120 --output nova_4d_double_integrator_syn/results_trial --no-plots
./.venv/bin/python -m nova_4d_double_integrator_syn verify \
    --output nova_4d_double_integrator_syn/results_trial --no-plots
```

Each `train` starts a fresh search and replaces files in its output directory;
it does not resume a checkpoint. Time limits are checked between operations.
Final verification, plotting, and an in-progress optimization call can extend
elapsed time beyond the requested search budget.

To change rollout display settings:

```bash
./.venv/bin/python -m nova_4d_double_integrator_syn animate --rollouts 8 --rollout-seed 10 \
    --rollout-dt .001 --rollout-seconds 20
```

These rollout flags also apply to train, verify, and plot. Simulation uses
Euler–Maruyama on the original SDE, stops at a sampled goal/unsafe/domain-exit
event or time limit, and displays the actual deployed control. Discrete-time
rollouts illustrate behavior; they do not establish the certificate guarantee.

Dependencies are managed by the repository's shared
[root requirements.txt](../requirements.txt). From the repository root:

```bash
python -m pip install -r requirements.txt
python nova_4d_double_integrator_syn/main.py train --no-plots
python nova_4d_double_integrator_syn/check.py nova_4d_double_integrator_syn/results/candidate.json
```

The source folder remains self-contained. For a standalone copy, install the
root requirements in its environment first; then run `python main.py train`
from the copied folder.

The last command needs only the Python standard library and verifies the problem
embedded in the candidate. The main CLI also checks that the selected problem
file matches the candidate, preventing reuse of an older SAT report after a
region change.

## 6. Evaluate the learned models in your own code

Both single states and batches are supported. `Controller.values` returns point
values of V and GV; it does not return interval bounds.

```python
import json
import numpy as np
from nova_4d_double_integrator_syn.runtime import Controller

with open("nova_4d_double_integrator_syn/results/candidate.json") as stream:
    candidate = json.load(stream)
policy = Controller(candidate)

state = np.array([-3.0, 1.5, 0.4, -1.07])  # [px, py, vx, vy]
u = policy(state)                         # shape (2,): [ux, uy]
V, GV = policy.values(state)               # two scalar point values

states = np.array([state, [0.1, 0.1, 0.0, 0.0]])
controls = policy(states)                 # shape (2, 2)
values, generators = policy.values(states) # each shape (2,)
```

The runtime loads coefficients and computes the internal coordinate
transformation. Supply original physical state coordinates in the specified
order; do not pre-normalize them or supply `qx, qy` instead.

## 7. Settings and limits of the search

Edit [problem.json](problem.json) for regions, acceleration-noise amplitudes,
control limits, beta, and epsilon. Use `--problem FILE` for an alternate
specification. The goal must contain the origin in its interior, the initial
`px` center must be nonzero, and unsafe boxes must cover the domain faces in
the form accepted by the validator.

Edit [search.json](search.json), or select another file with `--search FILE`:

| Setting | Default | Meaning |
|---|---|---|
| `curve_amplitudes` | `[1.2, 1.15, 1.1]` | Candidate bends, searched in order |
| `transverse_damping` | `[3.5, 4.5, 6.0]` | Candidate `dy` values for each bend |
| `longitudinal_damping` | `0.5` | Fixed `dx` used in proposals |
| `energy_caps` | `[11.8, 1.2]` | Cover caps for `qx, qy`; require certificate cap constraints |
| `grid_size` | `150` | Grid resolution per energy axis in proposal fitting |
| `goal_boundary_points` | `1500` | Extra proposal points near the goal enclosure boundary |
| `initial_training_upper` | `0.95` | Proposal target at initial corners |
| `generator_margin_factor` | `1.2` | Proposal target is `GV_upper <= -1.2*epsilon` |
| `refine_per_step` | `128` | Maximum selected splits per group per iteration |
| `max_cells` | `20000` | Separate cell cap for energy, initial, and unsafe groups |

The LP proposal also uses fixed factors `1.04*beta_ra` for interior obstacles
and `1.004*beta_ra` for energy-cap constraints. The refinement margins and scaling
rule described above are constants in `bound_training.py`, not entries in
`search.json`. `--seed` is recorded and sets the Torch seed; the present search
uses a fixed candidate order and deterministic grids rather than random restarts.

This is a restricted controller/certificate family specialized to the stated
stochastic double integrator. Changed specifications may need different search
ranges, gains, energy weights, caps, or a richer family. Arbitrary SDE changes
require corresponding generator and verifier changes. Search exhaustion returns
UNKNOWN, not a claim of infeasibility or guaranteed eventual convergence. After
selecting the first feasible screened proposal, the current implementation does
not return to the controller search if later refinement remains inconclusive.

## 8. Files and diagnostics

| File | Purpose |
|---|---|
| `main.py` | CLI and orchestration |
| `curved.py` | Certificate family and floating-point interval bounds |
| `bound_training.py` | Controller search, LP fitting, refinement, rescaling |
| `check.py`, `curved_check.py` | Independent certificate verification |
| `runtime.py` | Deployed controller, point V/GV, original-SDE rollouts |
| `visualize.py`, `animation.py`, `rollout_player.html` | Plots and offline animation |
| `results/candidate.json` | Full problem, model parameters, settings, search attempts, splits |
| `results/verification.json` | SAT/UNKNOWN, bound summaries, failed conditions, proof cells |
| `results/training.csv` | Iteration times, bound violations, and cell counts |
| `results/certificate_slices.pdf` | V/GV slices, contours, and region intersections |
| `results/rollouts.pdf`, `results/rollouts.json` | Simulation plots and recorded state/control samples |
| `results/rollouts_animation.html` | Interactive rollout and control-signal animation |

For an inconclusive run, inspect `failed_conditions`, `failed_generator_cells`,
`control_cover`, `training_stop`, and the last training-log rows. Saved
`controller_search` entries distinguish LP failures from control-cover failures.
Decimal summaries are for reading; rational bounds and comparisons determine SAT.

The legacy convex trainer remains in `synthesis.py` for reference. The current
CLI uses `bound_training.py`. `results_before_obstacle/` contains the archived
older experiment; development trials are not loaded by training.

Implementation checks can be run without synthesis:

```bash
./.venv/bin/python -m unittest nova_4d_double_integrator_syn.test_solver -v
```

They check interval coverage, generator derivatives, coefficient fitting,
controller bounds, original-SDE integration, and animation metadata. They do not
replace the independent verification of a trained candidate.
