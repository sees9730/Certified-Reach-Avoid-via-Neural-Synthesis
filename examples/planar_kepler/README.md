# Planar Kepler control synthesis

This example jointly synthesizes a bounded neural controller and a stochastic
reach-avoid certificate. It follows `examples/xv15_uncertain`: shared physics
in `model.py`, editable problem data in `config.json`, a
`neural_certified_nominal_drift/main.py` runner, per-seed outputs, and `_tests`.
Four-dimensional value/generator networks, rectangular discretization, and
plots use the shared framework also used by `examples/verification/4D_gbm_veri`.

## State and dynamics

The state order everywhere is **[r, theta, r_dot, theta_dot]**. The controls
**[u_r, u_t]** are radial and tangential acceleration, respectively. They act
only in the velocity equations. With gravitational parameter `mu = GM > 0`,

```text
dr         = r_dot dt
dtheta     = theta_dot dt
dr_dot     = (r theta_dot² - mu/r² + u_r) dt + sigma_r dW_r
dtheta_dot = ((u_t - 2 r_dot theta_dot)/r) dt + (sigma_t/r) dW_t
```

`W_r` and `W_t` are independent Brownian motions. The drift follows the
[polar acceleration equations](https://mechanicsmap.psu.edu/websites/9_newtons_particle/9-4_polar_equations_of_motion/polar_equations_of_motion.html).
Tangential acceleration is divided by `r` to obtain angular acceleration;
`u_t` is not torque or direct angular acceleration. For forces, divide by the
spacecraft mass before using these equations. There is no drag. Setting both
noise intensities to zero recovers deterministic controlled Kepler dynamics.

The diffusion is diagonal in the four-state representation:
`g(x) = [0, 0, sigma_r, sigma_t/r]`. Position has finite-variation paths in
this acceleration-noise model, so no additional Ito correction is needed.
This is a modeling choice; the example does not copy GBM noise to orbital
coordinates or assume Cartesian position noise.

The default uses normalized length/time units, `mu=1`, and radians. `theta`
is unwrapped in a finite angular sector of width less than `2*pi`; angle
wrapping is not applied in the certificate graph. The domain has `r_min>0`
and excludes the gravitational singularity. The implementation uses
`r_safe = r_min + relu(r-r_min)` to extend the graph to the bound engine's
zero dummy inputs. This equals `r` throughout the configured domain; states
outside the domain are failures in simulation, not a regularized orbital
model to be certified.

## Default reach-avoid task

The supplied configuration is an editable **station-keeping reach-avoid
example**, not a circular-orbit tracking task:

| Coordinate | Domain | Initial set | Goal set |
| --- | --- | --- | --- |
| r | [0.6, 1.6] | [1.2, 1.3] | [0.95, 1.05] |
| theta | [-0.6, 0.6] | [0.2, 0.3] | [-0.05, 0.05] |
| r_dot | [-0.6, 0.6] | [-0.15, -0.05] | [-0.05, 0.05] |
| theta_dot | [-0.6, 0.6] | [-0.15, -0.05] | [-0.05, 0.05] |

Unsafe sets are eight boundary slabs, one on each domain face, with thickness
0.05 along that face's coordinate. Thus radius, angle, and both velocity
limits are safety constraints. Generator cells cover the full safe interior
outside the goal. All membership decisions use all four coordinates.

Default accelerations are bounded by `u_r ∈ [-3,3]`, `u_t ∈ [-2,2]`, with
`sigma_r=sigma_t=0.02`, and `beta_ra=5`. At the goal center the stationary
equilibrium is `x_eq=[1,0,0,0]`, `u_eq=[1,0]`: outward radial acceleration
balances gravity. Configuration validation rejects impossible equilibrium
controls, nonpositive radius, invalid boxes, and goals that exclude zero
velocity. A circular orbit has nonzero `theta_dot` and changing `theta`;
orbit tracking would require a different goal/certificate formulation.

## Controller and certificate

The controller is a bias-free **4→64→2** tanh MLP. State errors are centered
at the stationary equilibrium and scaled per coordinate. A final tanh maps
outputs into the configured acceleration limits. The fixed tanh offset
preserves `u(x_eq)=u_eq` during training, as in the XV-15 trim controller.

The value network has four inputs and hidden widths 64 and 16, following the
4D GBM example. Its generator is
`GV = f·grad(V) + 0.5 sigma_r² V_r_dot,r_dot + 0.5 (sigma_t/r)² V_theta_dot,theta_dot`.
The runner first trains the value function on samples, then uses
the shared differentiable bound trainer with adaptive 4D refinement. Controller
updates begin when the bound-generator loss is activated. Every
four-dimensional bisection produces 16 child cells. No time or energy state
is added; nominal gravity is used.

The value MLP starts with eight smooth features, one for each unsafe domain
face, plus small random extra features. This provides initial safety separation
without changing the network architecture; all weights remain trainable.
Sample pretraining focuses on value constraints at learning rate 0.001.

Bound training prioritizes safety:

- At least 1,000 epochs optimize only the value constraints. Generator
  refinement and merging remain inactive during this phase.
- Generator weight then increases toward 1 over 3,000 safe epochs, provided
  the worst-cell violations of all value constraints are at most 0.1. Unsafe
  readiness uses the training target `1.01 * beta_ra`, including its buffer.
- When a value constraint regresses beyond that tolerance, generator weight
  is halved each epoch; recovery resumes a gradual ramp. This is an optimization
  schedule: final certification still requires the original exact checks.
- Bound losses are per-cell means, so adding cells does not mechanically
  multiply a uniform violation. Unsafe loss has weight 5; other value losses
  have weight 1. The bound learning rate is 0.001, with gradient norm capped
  at 1. Logged component losses are means; `Total` includes their weights.

The schedule addresses the original seed0 run's collapse toward `V=0.1`,
where coarse generator bounds dominated the initial updates and repeated
refinement inflated summed unsafe loss. It avoids treating a nearly constant
value function with small generator loss as a useful safety certificate.
`Generator weight` is recorded in the log and loss history. A zero logged
generator loss during warmup means it was not optimized, not that it passed.
The shared trainer retains summed losses and constant generator weight for
other examples unless these options are explicitly enabled.

## Run

From the repository root, using the existing environment:

```bash
./.venv/bin/python -u examples/planar_kepler/neural_certified_nominal_drift/main.py
```

Defaults: seed 0, CPU, one Torch thread, 5,000 sample epochs, 30,000 bound
epochs. Flags include `--seed N`, `--config PATH`, `--epochs N`,
`--pretrain-epochs N` (zero skips sample training), `--device cuda`,
`--threads N`, `--no-plots`, and `--output-dir PATH` (the run directory).
`--generator-warmup N` and `--generator-ramp N` override the bound schedule.
`--max-cells N` sets the refinement cell-count threshold for each of the
goal, init, unsafe, outside, and generator regions (default: `50000`).
`--outside-merge-margin VALUE` defaults to `10.0`; outside cells can merge
when their value lower bounds are at least this margin.
`--generator-merge-margin VALUE` defaults to `-1000.0`; generator cells can
merge when their generator upper bounds are at most this margin. The shared
trainer caps the generator threshold at `-1e-4` if a larger value is supplied.
Schedule settings, loss controls, cell-count thresholds, and merging margins
are saved in `outputs/run_config.json`. Evaluation-only runs reload these
saved refinement settings.

Start a tuned run in a separate directory to preserve the original seed0 log:

```bash
./.venv/bin/python -u examples/planar_kepler/neural_certified_nominal_drift/main.py \
    --no-plots \
    --output-dir examples/planar_kepler/neural_certified_nominal_drift/seed0_tuned
```

A short integration check, rather than a certification run:

```bash
./.venv/bin/python examples/planar_kepler/neural_certified_nominal_drift/main.py \
    --epochs 1 --pretrain-epochs 1 --no-plots \
    --output-dir /tmp/planar_kepler_smoke
```

Existing final bundles are protected from accidental training overwrites.
Use a new output directory for a new run. Reload and evaluate an existing run:

```bash
./.venv/bin/python examples/planar_kepler/neural_certified_nominal_drift/main.py \
    --evaluate-only --output-dir /tmp/planar_kepler_smoke
```

The default seed run is `neural_certified_nominal_drift/seed0/`, containing:

- `outputs/run_config.json`: the exact problem and hyperparameters.
- `outputs/V_pretrained.pth`, `controller_pretrained.pth`: sample-stage weights.
- `outputs/eval_bundle.pth`: final weights, region cells, history, and checks,
  saved even if training ends without satisfying all constraints.
- `outputs/certification_summary.json`: whether all five final bound checks
  passed. A saved checkpoint alone does not establish certification.
- `outputs/terminal_log.txt` and optionally `resume_checkpoint.pth`: training
  log and the shared trainer's SAT checkpoint.
- `results/`, `training_progress/`: shared 4D summary/progress plots. In those
  figures x1=r, x2=theta, x3=r_dot, and x4=theta_dot. Two-dimensional slices
  and region projections do not replace the four-dimensional bound checks.

## Empirical trajectories

After training, run the saved controller with reproducible acceleration noise:

```bash
./.venv/bin/python examples/planar_kepler/run_mc.py --n-mc 256 --n-paths 10
```

For the integration-check controller:

```bash
./.venv/bin/python examples/planar_kepler/run_mc.py \
    --checkpoint /tmp/planar_kepler_smoke/outputs/eval_bundle.pth \
    --output-dir /tmp/planar_kepler_mc
```

Additional flags: `--seed N`, `--dt VALUE`, `--t-max VALUE`,
`--deterministic` (zero simulation noise), and `--no-plots`. The script loads
the saved normalization/trim and training configuration. It stops each
trajectory on full-state goal entry, unsafe entry, domain exit, or a
nonfinite state. `run_mc_results/` contains a summary, cached trajectories,
four state-versus-time plots, and Cartesian orbital-plane position plots.
Euler-Maruyama success rates are empirical and can depend on step size.

## Checks

```bash
./.venv/bin/python -m pytest examples/planar_kepler/_tests -q
```

Tests verify polar-to-Cartesian force consistency, uncontrolled energy and
angular-momentum conservation, circular-orbit dynamics, acceleration-noise
scaling, controller limits/trim, analytic versus autograd generators,
differentiable bounds enclosing samples, complete safe 4D generator coverage,
configuration validation, full-state event checks, and rollout reproducibility.
