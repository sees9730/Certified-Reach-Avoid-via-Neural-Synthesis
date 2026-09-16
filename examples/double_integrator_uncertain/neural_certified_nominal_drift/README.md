# Nominal double-integrator certificate/controller synthesis

See [the example documentation](../README.md) and [config.json](../config.json)
for the dimensionless SDE, control limits and explicit reach-avoid boxes.

```bash
.venv/bin/python examples/asteroid_landing_uncertain/neural_certified_nominal_drift/main.py --seed 0 --no-plots
```

The default tag is `double_integrator`, preserving existing historical seed
folders. Use a fresh `--run-tag` when repeating an experiment.

Pipeline: sampled pretraining, joint interval-bound training, final full-cell
verification. `cell_grid.py` allocates the unsafe-cell budget to every boundary
strip. `unsat_diagnostics.py` exports failing full cells with `[px, py, vx, vy]`
endpoints and all six coordinate projections, even with `--no-plots`.
Projection rectangles are not fixed-coordinate slices.

Outputs include `run_config.json`, `terminal_log.txt`, pretrained network
weights, and the final `eval_bundle.pth`. The final bundle is written after
normal training completion even when SAT is not achieved. Results include
`unsat_cells/summary.json`, `unsat_cells.csv`, and plots for failing conditions.
Shared refinement behavior and cell limits come from `src/`; no additional
local training stage is used.

A small loss or a successful animation is not a certification result. Check
all full-cell SAT conditions in the final evaluation. No training or tests
were executed for this conversion.
