"""Full-cell UNSAT bounds, CSV export, and all six 2D projections for 4D."""
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from src.crown_bounds import SymbolicCROWNCache, SymbolicCROWNCache_Phi, prepare_cell_bounds
from src.training_utils import GENERATOR_MARGIN, generator_bound_masks

VALUE_GROUPS = ("goal", "init", "unsafe", "outside")
STATE_LABELS = ("px", "py", "vx", "vy")


@torch.no_grad()
def compute_diagnostic_bounds(value, generator, cells, params, batch_size=512):
    """Recompute bounds on the final partition/weights, in bounded-size batches.

    Refinement may have just replaced parents with children. Pre-refinement
    bounds are deliberately not reused for these diagnostic or SAT checks.
    """
    if batch_size < 1:
        raise ValueError("Diagnostic batch_size must be positive")
    value.eval()
    generator.eval()
    device, dim = params.training.device, params.network.n_inputs
    v_caches, phi_caches = {}, {}
    region_bounds = {}
    phi_parts = []
    for name in (*VALUE_GROUPS, "generator"):
        lower_parts, upper_parts = [], []
        for start in range(0, len(cells[name]), batch_size):
            chunk = cells[name][start:start + batch_size]
            n = len(chunk)
            lo, hi = prepare_cell_bounds(chunk, device, dim)
            if n not in v_caches:
                # Same bound methods as training, or these diagnostics would
                # judge SAT on bounds the loss never saw.
                v_caches[n] = SymbolicCROWNCache(value, n, dim, device,
                                                 method=params.training.bound_method)
            lb, ub = v_caches[n].compute_bounds(lo, hi)
            lower_parts.append(lb.detach().cpu())
            upper_parts.append(ub.detach().cpu())
            if name == "generator":
                if n not in phi_caches:
                    phi_caches[n] = SymbolicCROWNCache_Phi(generator, n, dim, device,
                                                           method=params.training.generator_bound_method)
                phi_parts.append(phi_caches[n].compute_bounds(lo, hi).detach().cpu())
        region_bounds[name] = (
            torch.cat(lower_parts) if lower_parts else torch.empty(0),
            torch.cat(upper_parts) if upper_parts else torch.empty(0),
        )
    return region_bounds, torch.cat(phi_parts) if phi_parts else torch.empty(0)


def unsat_masks(region_bounds, phi_upper, beta):
    """Use certificate thresholds, not the stronger training-loss targets."""
    masks, violations = {}, {}
    for name in VALUE_GROUPS:
        lb, ub = region_bounds[name]
        bound = ub if name == "init" else lb
        violation = ub - 1.0 if name == "init" else (beta if name == "unsafe" else 0.0) - lb
        masks[name] = (violation > 0) | ~torch.isfinite(bound)
        violations[name] = violation.clamp_min(0)
        violations[name][~torch.isfinite(bound)] = float("nan")
    active, masks["generator"] = generator_bound_masks(phi_upper, beta, region_bounds["generator"][0])
    violations["generator"] = (phi_upper + GENERATOR_MARGIN).clamp_min(0)
    # A nonfinite V bound is a failure even when the GV bound is finite.
    violations["generator"] = violations["generator"].clone()
    nonfinite = ~torch.isfinite(phi_upper) | ~torch.isfinite(region_bounds["generator"][0])
    violations["generator"][nonfinite] = float("nan")
    return masks, violations, active


def save_unsat_diagnostics(cells, regions, params, region_bounds, phi_upper, output_dir):
    """Export every failing cell and its projections; never just cell centers."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    masks, violations, active = unsat_masks(region_bounds, phi_upper, params.constraints.beta_ra)
    summary = dict(
        beta_ra=params.constraints.beta_ra, generator_margin=GENERATOR_MARGIN,
        active_generator_cells=int(active.sum().item()),
        groups={name: dict(total=len(cells[name]), unsat=int(mask.sum().item()))
                for name, mask in masks.items()},
        plot_note="Each rectangle is a projection of a full 4D cell, not a 2D slice. "
                  "A failed bound does not prove every point in the cell violates the condition.",
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)

    fields = ["condition", "cell_index"]
    fields += [f"{label}_{side}" for label in STATE_LABELS for side in ("lower", "upper")]
    fields += ["V_lower", "V_upper", "GV_upper", "generator_active", "bound_violation"]
    with (output_dir / "unsat_cells.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, mask in masks.items():
            lb, ub = region_bounds[name]
            for i in torch.nonzero(mask, as_tuple=True)[0].tolist():
                lo, hi = cells[name][i]
                row = dict(condition=name, cell_index=i, V_lower=lb[i].item(), V_upper=ub[i].item(),
                           GV_upper=phi_upper[i].item() if name == "generator" else "",
                           generator_active=bool(active[i].item()) if name == "generator" else "",
                           bound_violation=violations[name][i].item())
                row.update({f"{label}_lower": lo[d].item() for d, label in enumerate(STATE_LABELS)})
                row.update({f"{label}_upper": hi[d].item() for d, label in enumerate(STATE_LABELS)})
                writer.writerow(row)

    for name, mask in masks.items():
        indices = torch.nonzero(mask, as_tuple=True)[0].tolist()
        print(f"UNSAT diagnostics: {name}: {len(indices)}/{len(cells[name])} cells")
        if indices:
            lo = np.stack([cells[name][i][0].detach().cpu().numpy() for i in indices])
            hi = np.stack([cells[name][i][1].detach().cpu().numpy() for i in indices])
            _plot_projections(name, lo, hi, violations[name][mask].numpy(), regions, output_dir)
    print(f"Saved full-cell UNSAT diagnostics -> {output_dir}")


def _plot_projections(name, lo, hi, severity, regions, output_dir):
    # An explicit Agg canvas saves images without opening GUI windows.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch, Rectangle
    from matplotlib import colormaps

    fig = Figure(figsize=(15, 9), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 3)
    finite = np.isfinite(severity)
    vmax = max(float(severity[finite].max()) if finite.any() else 0.0, 1e-8)
    norm = Normalize(vmin=0, vmax=vmax)
    colors = colormaps["YlOrRd"](norm(np.where(finite, severity, vmax)))
    colors[~finite] = (0, 0, 0, 1)
    # Draw the largest violations last so that projections remain visible.
    order = np.argsort(np.where(finite, severity, np.inf), kind="stable")
    full = regions.full.bounds

    def draw_region(ax, region, a, b, color):
        if region.is_union:
            for component in region.components:
                draw_region(ax, component, a, b, color)
        else:
            bounds = region.bounds
            ax.add_patch(Rectangle((bounds[a, 0], bounds[b, 0]),
                                   bounds[a, 1] - bounds[a, 0], bounds[b, 1] - bounds[b, 0],
                                   fill=False, edgecolor=color, linewidth=1.5, linestyle="--", zorder=3))

    for ax, (a, b) in zip(axes.flat, combinations(range(4), 2)):
        vertices = np.stack([
            np.column_stack([lo[:, a], lo[:, b]]), np.column_stack([hi[:, a], lo[:, b]]),
            np.column_stack([hi[:, a], hi[:, b]]), np.column_stack([lo[:, a], hi[:, b]]),
        ], axis=1)
        collection = PolyCollection(vertices[order], facecolors=colors[order],
                                    edgecolors=colors[order], linewidths=0.3, alpha=0.55)
        ax.add_collection(collection)
        draw_region(ax, regions.init, a, b, "blue")
        draw_region(ax, regions.goal, a, b, "green")
        ax.set(xlim=full[a], ylim=full[b], xlabel=STATE_LABELS[a], ylabel=STATE_LABELS[b])
        ax.grid(alpha=0.15)
    from matplotlib.cm import ScalarMappable
    fig.colorbar(ScalarMappable(norm=norm, cmap="YlOrRd"), ax=axes.ravel().tolist(),
                 label="Bound violation (black = nonfinite bound)", shrink=0.85)
    axes[0, 0].legend(handles=[Patch(fill=False, edgecolor="blue", label="Initial projection"),
                               Patch(fill=False, edgecolor="green", label="Goal projection")], fontsize=8)
    fig.suptitle(f"{name}: {len(lo)} UNSAT cells — all six 4D-to-2D projections\n"
                 "Rectangles show full cell extents; overlapping projections are not 2D slices.")
    fig.savefig(output_dir / f"{name}_unsat_projections.png", dpi=180)
    fig.clear()
