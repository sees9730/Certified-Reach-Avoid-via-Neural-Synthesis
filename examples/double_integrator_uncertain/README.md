# Nominal 4D double-integrator SDE

The existing folder name is retained, but the active example is now a double
integrator. States, controls and time are dimensionless; no unit conversion
is performed.

```text
state = [px, py, vx, vy], control = [ux, uy]
dpx = vx dt
dpy = vy dt
dvx = ux dt + sqrt(qx) dWx
dvy = uy dt + sqrt(qy) dWy
```

Wx and Wy are independent standard Brownian motions. The configured velocity
covariance rates are qx=qy=1.234e-7. Parameters are fixed; uncertainty synthesis
is not included in this first case. The generator is
`GV = vx V_px + vy V_py + ux V_vx + uy V_vy + (qx V_vxvx + qy V_vyvy)/2`.

## Problem configuration

`config.json` defines explicit boxes in `[px, py, vx, vy]` order:

| Region | px | py | vx | vy |
| --- | --- | --- | --- | --- |
| Full domain | [-1, 1] | [-1, 1] | [-1, 1] | [-1, 1] |
| Initial | [0.45, 0.55] | [0.15, 0.25] | [-0.025, 0.025] | [-0.025, 0.025] |
| Goal | [-0.15, 0.15] | [-0.15, 0.15] | [-0.15, 0.15] | [-0.15, 0.15] |

Unsafe regions are eight boundary strips, each 4% of its domain width. Thus
any coordinate at or beyond +/-0.92 within the full domain is unsafe.
The goal requires all four coordinates to lie in their goal intervals.
The neural controller has independent bounds `|ux| <= 1`, `|uy| <= 1`, read
from `control.u_max`; learned biases allow a nonzero command at the origin.

The certificate uses two sigmoid hidden layers and the existing analytic GV
implementation. `V(goal_center)=0.1` is anchored, while unsafe separation is
learned. The current beta is 4; unsafe training targets 4.04, but SAT requires
4. Active generator cells (`V_lower <= beta`) require `GV_upper <= -1e-4`.

## Training

From the repository root:

```bash
.venv/bin/python examples/asteroid_landing_uncertain/neural_certified_nominal_drift/main.py --seed 0 --no-plots
```

Default outputs go to `neural_certified_nominal_drift/seed0_double_integrator/`.
Use `--run-tag another_name` for another experiment. Reusing the same seed/tag
clears that run's output directories. Existing Kepler outputs are preserved.

The pipeline is sampled pretraining followed directly by joint full-cell
bound training of V and the controller, then final verification and UNSAT
cell diagnostics. The existing shared loss and refinement pipeline is kept;
there is no local warmup or alternating update stage. Training settings are
under `learning`, including learning rates, budgets and refinement frequency.
The runtime checks compare analytical GV against autograd, including the
origin. The origin is a regular state of these dynamics.

## Animation

Without training, use the saturated linear feedback baseline:

```bash
.venv/bin/python examples/asteroid_landing_uncertain/animate.py --controller baseline --kp 2 --kd 3
```

This uses `u = clip(-kp*(p-p_goal) - kd*(v-v_goal), -u_max, u_max)` and writes
`results/double_integrator_baseline.html`. Noise is enabled by default;
add `--deterministic` to disable it. The baseline has no neural certificate.

After a run writes `outputs/eval_bundle.pth`:

```bash
.venv/bin/python examples/asteroid_landing_uncertain/animate.py --seed 0
```

For a custom run, pass `--run-dir path/to/seed_folder`. The output is
`results/double_integrator_rollout.html`, a standalone position animation
with state, control and V/GV values. It reconstructs exactly the trained
model and stops on goal, unsafe, domain exit or time limit. Events are checked
in 4D at numerical integration steps. An individual rollout is not a proof.

The old gravitational drift, cone geometry, covariance-to-region setup,
unused certificate alternatives, hand-designed orbital baseline and orbital
HTML template have been removed from active code. Historical output files
and `config_before_v2_restore.json` are archives, not valid configs for this
model. Loaders reject configs without `problem: double_integrator_4d` so old
Kepler controllers cannot silently be presented as double-integrator results.

Training and regression tests have not been run for this conversion; SAT and
convergence remain to be checked in your run.
