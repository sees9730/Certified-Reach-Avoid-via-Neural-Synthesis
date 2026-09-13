# XV-15 synthesis with uncertain air density

This experiment uses **rho in [1.16375, 1.28625] kg/m³**, ±5% around the
nominal 1.225 kg/m³. The interval is in
`../config.json`, under `uncertainty.air_density_kg_m3`.

It reuses the nominal XV-15 neural architectures, regions, diffusion, trim
anchor, training settings, and refinement schedule. Both neural networks
start from scratch. The controller sees only `[v, gamma, beta]`; it does not
observe density. All other physical parameters remain nominal.

## Run

From the repository root:

```bash
./.venv/bin/python -u examples/xv15_uncertain/neural_certified_uncertain_param/main.py
```

Defaults match the nominal baseline: seed 0, 15,000 sample pretraining
epochs, 30,000 bound-training epochs, and `beta_ra = 5`. Optional flags are
`--seed`, `--device`, `--pretrain-epochs`, `--epochs`, `--no-plots`, and
`--config`, as in the nominal entry point. To override density without
editing the configuration:

```bash
./.venv/bin/python -u examples/xv15_uncertain/neural_certified_uncertain_param/main.py \
    --density-min 1.16375 --density-max 1.28625 --no-plots
```

Setting both endpoints to `1.225` recovers nominal generator conditions.
Endpoints must be finite and satisfy `0 < lower <= upper`.

Outputs are local to this experiment under `seed0/outputs/`,
`seed0/results/`, and `seed0/training_progress/`. Each run clears these
directories for its selected seed, following the nominal baseline.
Nominal-run outputs are in a separate directory.

`outputs/terminal_log.txt` contains the interval, settings, generator
check, training progress, and final bound summary. `run_config.json`
records the effective interval, including CLI overrides. Pretraining saves
both networks; final evaluation saves `eval_bundle.pth` even when the epoch
budget ends without SAT. The generator state dictionary also stores the
density interval, midpoint offset, and radius used in the support term.

## Robust generator

Lift `L` and drag `D` depend linearly on density. Let `L0,D0` be their values
at nominal density `rho0`. With a fixed state-feedback controller:

```text
f(x,u,rho) = f(x,u,rho0) + (rho-rho0) * a(x,u)
a(x,u)    = [-D0/(mass*rho0), L0/(mass*v*rho0), 0]
```

For midpoint `rho_c`, half-width `rho_r`, and direction `p`, the exact
support function of this line segment is

```text
max_rho p·f(x,u,rho)
    = p·f(x,u,rho0) + (rho_c-rho0)*(p·a) + rho_r*abs(p·a).
```

The absolute value is taken **after** summing the lift/drag contributions:
one shared density value changes both forces. This preserves the correlation
that would be lost by replacing the two drift components with independent
disturbances.

`DensityIntervalDrift.support(x,p)` implements this expression. The shared
`create_GV` detects that method, as it does for the robust pendulum, and uses
`p = grad(V)` before adding the unchanged Itô diffusion contribution:

```text
G_robust V = max_rho (grad(V)·f(x,u,rho))
             + 0.5 * sum_i sigma_i^2 * V_xi_xi.
```

This robust generator is used during sample pretraining, bound training,
final evaluation, and plotting. Bounds apply to the complete density
interval at each state; density is not sampled during training or added as
a state coordinate. The condition is uniform over density, so it covers a
fixed unknown density and also density varying within the interval.

`forward(x)` on the drift returns nominal dynamics for diagnostics; robust
certification uses `support`. The controller's equilibrium anchor is only
a nominal trim point, not an equilibrium asserted for every density.

## Implementation checks

```bash
MPLBACKEND=Agg ./.venv/bin/python -m pytest examples/xv15_uncertain/test_uncertain_density.py -q
```

These checks compare support values with physical endpoint drifts, check
correlated lift/drag cancellation and singleton intervals, compare the
robust generator with independent endpoint autograd calculations, and
exercise interval bounds and gradients through both networks. They do not
run synthesis or establish SAT for a trained candidate.
