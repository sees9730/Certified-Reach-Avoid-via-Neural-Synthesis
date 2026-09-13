# XV-15 synthesis with uncertain air density and mass

This experiment uses independent density and mass intervals from
`../config.json`, under `uncertainty.air_density_kg_m3` and
`uncertainty.mass_kg`. Density **[1.16375, 1.28625] kg/m³** is ±5% around
the nominal 1.225 kg/m³. Mass **[5841, 5959] kg** is ±1% around the nominal
5900 kg; use the explicit command below to select this interval regardless
of the current configuration.

It reuses the nominal XV-15 neural architectures, regions, diffusion, trim
anchor, training settings, and refinement schedule. Both neural networks
start from scratch. The controller sees only `[v, gamma, beta]`; it does not
observe density or mass. Thrust limits and the trim anchor use nominal mass;
the same commanded thrust in newtons is applied at every uncertain mass.
All other physical parameters remain nominal.

## Run

From the repository root:

```bash
./.venv/bin/python -u examples/xv15_uncertain/neural_certified_uncertain_param/main.py
```

Defaults match the nominal baseline: seed 0, 15,000 sample pretraining
epochs, 30,000 bound-training epochs, and `beta_ra = 5`. Optional flags are
`--seed`, `--device`, `--pretrain-epochs`, `--epochs`, `--no-plots`, and
`--config`, as in the nominal entry point. To override the intervals without
editing the configuration:

```bash
./.venv/bin/python -u examples/xv15_uncertain/neural_certified_uncertain_param/main.py \
    --density-min 1.16375 --density-max 1.28625 \
    --mass-min 5841 --mass-max 5959 --no-plots
```

Setting both density endpoints to `1.225` and both mass endpoints to `5900`
recovers nominal generator conditions. Setting only the mass endpoints to
`5900` recovers density-only robustness. Endpoints must be finite and satisfy
`0 < lower <= upper`. Older configurations without `mass_kg` retain nominal mass.

Outputs are local to this experiment under `seed0/outputs/`,
`seed0/results/`, and `seed0/training_progress/`. Each run clears these
directories for its selected seed, following the nominal baseline.
Nominal-run outputs are in a separate directory.

`outputs/terminal_log.txt` contains both intervals, settings, generator
check, training progress, and final bound summary. `run_config.json`
records the effective intervals, including CLI overrides. Pretraining saves
both networks; final evaluation saves `eval_bundle.pth` even when the epoch
budget ends without SAT. The generator state dictionary also stores the
density and mass intervals and the constants used in the support term.

## Robust generator

Lift `L` and drag `D` depend linearly on density. Let `L0,D0` be their values
at nominal density `rho0`, and let `p` be a direction. With a fixed
state-feedback controller, the drift projection is

```text
p·f(x,u,rho,m) = c + (a + rho*b)/m
a = T * (cos(alpha+beta)*p_v + sin(alpha+beta)*p_gamma/v)
b = (-D0*p_v + L0*p_gamma/v)/rho0
c = -g * (sin(gamma)*p_v + cos(gamma)*p_gamma/v) + delta*p_beta
```

Gravity is independent of mass because weight divided by mass is `g`.
For density midpoint `rho_c` and half-width `rho_r`, first maximize the
force projection over density. Since mass is positive, the maximizing
density is independent of mass. Then maximize over reciprocal mass, with
midpoint `q_c = (1/m_min + 1/m_max)/2` and half-width
`q_r = (1/m_min - 1/m_max)/2`:

```text
F = a + rho_c*b + rho_r*abs(b)
max_rho,m p·f(x,u,rho,m) = c + q_c*F + q_r*abs(F).
```

The absolute value is taken **after** summing the lift/drag contributions:
one shared density value changes both forces, and one shared mass divides
both force accelerations. This expression equals the maximum over the four
physical density/mass corner combinations and covers the entire rectangle.

`DensityMassIntervalDrift.support(x,p)` in `../uncertain_parameters.py`
implements three equivalent expressions and takes their pointwise minimum
to tighten interval bounds, as described below. The shared
`create_GV` detects that method, as it does for the robust pendulum, and uses
`p = grad(V)` before adding the unchanged Itô diffusion contribution:

```text
G_robust V = max_rho,m (grad(V)·f(x,u,rho,m))
             + 0.5 * sum_i sigma_i^2 * V_xi_xi.
```

This robust generator is used during sample pretraining, bound training,
final evaluation, and plotting. Bounds apply to both complete parameter
intervals at each state; neither parameter is sampled during training or
added as a state coordinate. The condition is uniform over both parameters,
covering fixed unknown values and values varying within these intervals.

`forward(x)` on the drift returns nominal dynamics for diagnostics; robust
certification uses `support`. The controller's equilibrium anchor is only
a nominal trim point, not an equilibrium asserted for every density/mass pair.

### Tightening the state-cell bounds

Exact parameter maximization does not by itself make state-cell interval
bounds tight: algebraically equal expressions can lose different dependencies
during IBP. The drift therefore shares one controller evaluation and combines
three exact support formulas:

1. The decomposed expression `c + q_c*F + q_r*abs(F)` above.
2. A nominal-centered expression that retains the original density-only graph
   `S_rho = p·f(x,u,rho0,m0) + (rho_c-rho0)*b/m0 + rho_r*abs(b)/m0`, then
   adds only the mass correction:
   `S_centered = S_rho + (q_c-1/m0)*F + q_r*abs(F)`.
3. The maximum of four physical drift projections, one for each density/mass
   endpoint pair. Each projection retains force/gravity cancellation inside
   each acceleration before multiplication by `p`.

The corner reduction is exact because `c + q*a + rho*q*b` is multi-affine
in `(rho,q)`, where `q=1/m`. Its maximum over a rectangle occurs at a vertex.
The product `rho*q` must not be replaced by an independent uncertain parameter.
This is an analytical elimination of parameter uncertainty, consistent with
the closed-form generator approach; no parameter sampling or interior grid
approximation is used.

Native `torch.maximum` computes the corner maximum; native `torch.minimum`
combines the three equivalent support graphs. All three graphs equal the
same worst-case drift at a point. Under IBP their combination yields

```text
U_robust(cell) = min(U_decomposed(cell), U_centered(cell), U_corners(cell)),
```

including the common diffusion upper bound in each `U`. The minimum remains
a sound upper bound and is no looser than any of the three individual bounds
(up to floating-point roundoff). The maximum over parameter corners stays
inside `U_corners`; taking a minimum over parameter cases would be incorrect.
The certificate gradient/Hessian and the controller are shared across the
formulas. Bounds remain differentiable almost everywhere, allowing both neural
networks to train through the selected bounds.

In a 32-cell CPU diagnostic with untrained seed-0 networks, ±5% density, and
±1% mass, the initial-region upper bound decreases from approximately
**0.39843 to 0.27170**. Bound computation plus backpropagation took about 30%
longer in that diagnostic. This is a fixed-network comparison, not a completed
synthesis result or a prediction of total training time. State-cell IBP
can still be conservative, and the existing adaptive state refinement remains
necessary. There is no claim of a globally optimal upper bound or guaranteed SAT.

## Implementation checks

```bash
MPLBACKEND=Agg ./.venv/bin/python -m pytest \
    examples/xv15_uncertain/test_uncertain_density.py \
    examples/xv15_uncertain/test_uncertain_mass.py -q
```

These checks compare support values with physical corner drifts and interior
parameter values, check correlated lift/drag cancellation and singleton
intervals, compare the robust generator with independent corner autograd calculations, and
exercise interval bounds and gradients through both networks. Regression checks
also verify that the combined bound selects the tightest formula per cell and
recovers the pointwise generator on zero-width cells. They do not
run synthesis or establish SAT for a trained candidate.
