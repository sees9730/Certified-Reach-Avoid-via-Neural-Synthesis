"""
Visualization utilities for plotting value functions and generators.

This module provides functions to:
- Plot value function V(x) over the state space
- Plot generator GV(x) over the state space
- Overlay regions (init, goal, unsafe)
- Show discretization cells
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Optional, List, Tuple

# Set up directories
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]   # repo_root
import sys
sys.path.insert(0, str(ROOT))
from src.regions import Regions


def _get_plot_pairs(D: int):
    """
    Return list of (x_dim, y_dim) index pairs (0-based) following your rule:
      D=2: (1 vs 2)
      D=3: (1 vs 2), (3 vs 2)
      D=4: (1 vs 2), (3 vs 4)
      D=5: (1 vs 2), (3 vs 4), (5 vs 4)
    General rule:
      - Consecutive pairs: (1 vs 2), (3 vs 4), (5 vs 6), ...
      - If D is odd and > 2: add last dim vs previous dim: (D vs D-1)
    """
    if D <= 1:
        raise ValueError(f"D must be > 1, got D={D}")

    pairs = []
    # consecutive pairs: (0,1), (2,3), (4,5), ...
    for i in range(0, D - 1, 2):
        pairs.append((i, i + 1))

    # odd leftover: add (D-1) vs (D-2) => x_D vs x_{D-1}
    if (D % 2 == 1) and (D > 2):
        pairs.append((D - 1, D - 2))  # x-axis: last dim, y-axis: previous dim

    return pairs


def _format_dim_label(k_0based: int) -> str:
    """Pretty label like x₁, x₂, ... for small indices; falls back to x{n}."""
    n = k_0based + 1
    subs = str(n).translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))
    return f"x{subs}"


def _make_slice_grid(full_bounds: np.ndarray, x_dim: int, y_dim: int, resolution: int, slice_point: np.ndarray = None):
    """
    Build a (resolution^2, D) grid where only x_dim and y_dim vary over their bounds.
    All other dims are fixed at `slice_point` (if provided) or the center of full_bounds.
    Returns:
        X (mesh), Y (mesh), grid_np (N, D)
    """
    D = full_bounds.shape[0]

    x_vals = np.linspace(full_bounds[x_dim, 0], full_bounds[x_dim, 1], resolution)
    y_vals = np.linspace(full_bounds[y_dim, 0], full_bounds[y_dim, 1], resolution)
    X, Y = np.meshgrid(x_vals, y_vals)

    if slice_point is None:
        slice_point = 0.5 * (full_bounds[:, 0] + full_bounds[:, 1])  # (D,)
    else:
        slice_point = np.asarray(slice_point, dtype=np.float32)
        assert slice_point.shape == (D,), f"slice_point must be shape ({D},), got {slice_point.shape}"

    N = X.size
    grid = np.tile(slice_point[None, :], (N, 1))     # (N,D)
    grid[:, x_dim] = X.reshape(-1)
    grid[:, y_dim] = Y.reshape(-1)
    return X, Y, grid


def make_slice_point_for_region(
    region,
    full_bounds: np.ndarray,
    x_dim: int,
    y_dim: int,
    *,
    fallback: str = "center_full",   # or "center_region"
) -> np.ndarray:
    """
    Create a (D,) slice_point that passes through the middle of `region`
    for all dimensions except x_dim and y_dim.

    - For dims not in {x_dim, y_dim}: use center of the region bounds.
    - For x_dim and y_dim: keep fallback center (doesn't matter; those dims vary).
    - Clips result to full_bounds.

    Handles union regions by using the center of the union bounding box.
    """
    D = full_bounds.shape[0]

    # Base point (fallback for plotted dims)
    if fallback == "center_region":
        # If you want even plotted dims to start from region center (not necessary)
        base = 0.5 * (_region_bbox(region, D)[:, 0] + _region_bbox(region, D)[:, 1])
    else:
        base = 0.5 * (full_bounds[:, 0] + full_bounds[:, 1])

    region_bounds = _region_bbox(region, D)  # (D,2)
    region_center = 0.5 * (region_bounds[:, 0] + region_bounds[:, 1])

    slice_point = base.copy()
    for d in range(D):
        if d != x_dim and d != y_dim:
            slice_point[d] = region_center[d]

    # Safety: clip to full bounds
    slice_point = np.clip(slice_point, full_bounds[:, 0], full_bounds[:, 1]).astype(np.float32)
    return slice_point


def _region_bbox(region, D: int) -> np.ndarray:
    """
    Return a (D,2) bounding box for Region or union Region.
    Assumes `region.bounds` exists for non-union, and `region.components` for union.
    """
    if getattr(region, "is_union", False):
        lows = []
        highs = []
        for comp in region.components:
            b = np.asarray(comp.bounds, dtype=np.float32)
            lows.append(b[:, 0])
            highs.append(b[:, 1])
        low = np.min(np.stack(lows, axis=0), axis=0)
        high = np.max(np.stack(highs, axis=0), axis=0)
        return np.stack([low, high], axis=1)
    else:
        b = np.asarray(region.bounds, dtype=np.float32)
        if b.shape != (D, 2):
            raise ValueError(f"Region bounds shape {b.shape} != ({D},2)")
        return b


def _draw_region_proj(ax, region, color: str, label: str, x_dim: int, y_dim: int):
    """Draw region projected to (x_dim, y_dim) on the given axis."""
    if region.is_union:
        for i, comp in enumerate(region.components):
            bounds = comp.bounds
            comp_label = label if i == 0 else None
            rect = Rectangle(
                (bounds[x_dim, 0], bounds[y_dim, 0]),
                bounds[x_dim, 1] - bounds[x_dim, 0],
                bounds[y_dim, 1] - bounds[y_dim, 0],
                linewidth=3, edgecolor=color, facecolor='none', label=comp_label
            )
            ax.add_patch(rect)
    else:
        bounds = region.bounds
        rect = Rectangle(
            (bounds[x_dim, 0], bounds[y_dim, 0]),
            bounds[x_dim, 1] - bounds[x_dim, 0],
            bounds[y_dim, 1] - bounds[y_dim, 0],
            linewidth=3, edgecolor=color, facecolor='none', label=label
        )
        ax.add_patch(rect)


def visualize_value_function(
    V_net,
    regions: Regions,
    show_regions: bool = True,
    show_discretization: bool = False,
    epoch: Optional[int] = None,
    training_cells: Optional[List] = None,
    filename: Optional[str] = None,
    resolution: int = 100,
    figsize: Tuple[int, int] = (10, 8)
):
    V_net.eval()

    full_bounds = regions.full.bounds
    D = full_bounds.shape[0]
    pairs = _get_plot_pairs(D)

    # device for eval
    try:
        net_device = next(V_net.parameters()).device
    except StopIteration:
        net_device = torch.device("cpu")

    n_plots = len(pairs)
    fig, axes = plt.subplots(
        1, n_plots,
        figsize=(figsize[0] * n_plots, figsize[1]),
        squeeze=False
    )
    axes = axes.ravel()

    for p, (x_dim, y_dim) in enumerate(pairs):
        ax = axes[p]

        X, Y, grid_np = _make_slice_grid(full_bounds, x_dim, y_dim, resolution)
        x_grid = torch.tensor(grid_np, dtype=torch.float32, device=net_device)

        with torch.no_grad():
            V_output = V_net(x_grid).detach().cpu().numpy().reshape(X.shape)

        vmin, vmax = float(np.min(V_output)), float(np.max(V_output))

        contour = ax.contourf(X, Y, V_output, levels=20, cmap='viridis')
        fig.colorbar(contour, ax=ax, label=f"V({_format_dim_label(x_dim)}, {_format_dim_label(y_dim)})")

        ax.set_xlabel(_format_dim_label(x_dim), fontsize=12)
        ax.set_ylabel(_format_dim_label(y_dim), fontsize=12)
        if epoch is not None:
            ax.set_title(f"{_format_dim_label(x_dim)} vs {_format_dim_label(y_dim)}\n"
                        f"Epoch: {epoch} - V(x) Output Range: [{vmin:.4f}, {vmax:.4f}]",
                        fontsize=12, fontweight='bold')
        else:
            ax.set_title(f"{_format_dim_label(x_dim)} vs {_format_dim_label(y_dim)}\n"
                        f"Final Evaluation - V(x) Output Range: [{vmin:.4f}, {vmax:.4f}]",
                        fontsize=12, fontweight='bold')

        # Regions (projected)
        if show_regions:
            # only label on first subplot to avoid legend spam
            lab_init  = "Init" if p == 0 else None
            lab_unsafe = "Unsafe" if p == 0 else None
            lab_goal  = "Goal" if p == 0 else None

            _draw_region_proj(ax, regions.init, 'seagreen', lab_init, x_dim, y_dim)
            _draw_region_proj(ax, regions.unsafe, 'firebrick', lab_unsafe, x_dim, y_dim)
            _draw_region_proj(ax, regions.goal, 'darkgoldenrod', lab_goal, x_dim, y_dim)

            if p == 0:
                ax.legend(loc='upper right', fontsize=10)

        # Discretization cells (projected)
        if show_discretization and training_cells is not None:
            for cell_lower, cell_upper in training_cells:
                rect = Rectangle(
                    (cell_lower[x_dim].item(), cell_lower[y_dim].item()),
                    cell_upper[x_dim].item() - cell_lower[x_dim].item(),
                    cell_upper[y_dim].item() - cell_lower[y_dim].item(),
                    linewidth=0.5, edgecolor='dimgray', facecolor='none', alpha=1.0
                )
                ax.add_patch(rect)

    fig.tight_layout()

    if filename is None:
        filename = "value_function.pdf"
    fig.savefig(filename, dpi=500, format='pdf')
    plt.close(fig)

    V_net.train()


def visualize_generator(
    V_net,
    GV_net,
    regions: Regions,
    show_regions: bool = True,
    show_discretization: bool = False,
    epoch: Optional[int] = None,
    training_cells: Optional[List] = None,
    results=None,
    filename: Optional[str] = None,
    resolution: int = 100,
    figsize: Tuple[int, int] = (10, 8)
):
    V_net.eval()
    GV_net.eval()

    full_bounds = regions.full.bounds
    D = full_bounds.shape[0]
    pairs = _get_plot_pairs(D)

    try:
        net_device = next(GV_net.parameters()).device
    except StopIteration:
        net_device = torch.device("cpu")

    n_plots = len(pairs)
    fig, axes = plt.subplots(
        1, n_plots,
        figsize=(figsize[0] * n_plots, figsize[1]),
        squeeze=False
    )
    axes = axes.ravel()

    for p, (x_dim, y_dim) in enumerate(pairs):
        ax = axes[p]

        slice_point = make_slice_point_for_region(
            regions.goal,
            full_bounds,
            x_dim=x_dim,
            y_dim=y_dim,
        )

        X, Y, grid_np = _make_slice_grid(full_bounds, x_dim, y_dim, resolution, slice_point=slice_point)
        x_grid = torch.tensor(grid_np, dtype=torch.float32, device=net_device)

        with torch.no_grad():
            Phi_output = GV_net(x_grid).detach().cpu().numpy().reshape(X.shape)

        phi_min = float(np.min(Phi_output))
        phi_max = float(np.max(Phi_output))
        if results is not None:
            # keep your old behavior: show global min/max if provided
            phi_min = float(results.get('Phi_min', phi_min))
            phi_max = float(results.get('Phi_max', phi_max))

        abs_max = max(abs(phi_min), abs(phi_max))
        contour = ax.contourf(X, Y, Phi_output, levels=20, cmap='RdBu_r',
                              vmin=-abs_max, vmax=abs_max)
        fig.colorbar(contour, ax=ax, label=f"GV({_format_dim_label(x_dim)}, {_format_dim_label(y_dim)})")

        # GV=0 contour
        ax.contour(X, Y, Phi_output, levels=[0], colors='black', linewidths=2)

        ax.set_xlabel(_format_dim_label(x_dim), fontsize=12)
        ax.set_ylabel(_format_dim_label(y_dim), fontsize=12)
        if epoch is not None:
            ax.set_title(f"{_format_dim_label(x_dim)} vs {_format_dim_label(y_dim)}\n"
                     f"Epoch {epoch} - GV(x) Output Range: [{phi_min:.4f}, {phi_max:.4f}]",
                     fontsize=12, fontweight='bold')
        else:
            ax.set_title(f"{_format_dim_label(x_dim)} vs {_format_dim_label(y_dim)}\n"
                     f"Final Evaluation - GV(x) Output Range: [{phi_min:.4f}, {phi_max:.4f}]",
                     fontsize=12, fontweight='bold')

        if show_regions:
            lab_init  = "Init" if p == 0 else None
            lab_unsafe = "Unsafe" if p == 0 else None
            lab_goal  = "Goal" if p == 0 else None

            _draw_region_proj(ax, regions.init, 'seagreen', lab_init, x_dim, y_dim)
            _draw_region_proj(ax, regions.unsafe, 'firebrick', lab_unsafe, x_dim, y_dim)
            _draw_region_proj(ax, regions.goal, 'darkgoldenrod', lab_goal, x_dim, y_dim)

            if p == 0:
                ax.legend(loc='upper right', fontsize=10)

        if show_discretization and training_cells is not None:
            for cell_lower, cell_upper in training_cells:
                rect = Rectangle(
                    (cell_lower[x_dim].item(), cell_lower[y_dim].item()),
                    cell_upper[x_dim].item() - cell_lower[x_dim].item(),
                    cell_upper[y_dim].item() - cell_lower[y_dim].item(),
                    linewidth=0.5, edgecolor='dimgray', facecolor='none', alpha=1.0
                )
                ax.add_patch(rect)

    fig.tight_layout()

    if filename is None:
        filename = "generator.pdf"
    fig.savefig(filename, dpi=500, format='pdf')
    plt.close(fig)

    V_net.train()
    GV_net.train()


def visualize_training_progress(
    V_net,
    GV_net,
    regions: Regions,
    region_cells: dict,
    epoch: int,
    output_dir: str = "."
):
    """
    Visualize training progress (both V and GV) with discretization.

    Args:
        V_net: Value function network
        GV_net: Generator network
        regions: Regions object
        region_cells: Dictionary of region cells
        epoch: Current epoch number
        output_dir: Directory to save plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Cells for V network (all regions except generator)
    v_cells = []
    for region_name in ['init', 'goal', 'unsafe', 'outside']:
        if region_name in region_cells:
            v_cells.extend(region_cells[region_name])

    # Cells for generator network (only generator region)
    gv_cells = region_cells.get('generator', [])

    # Plot value function with V discretization
    visualize_value_function(
        V_net=V_net,
        regions=regions,
        epoch=epoch,
        show_regions=True,
        show_discretization=True,
        training_cells=v_cells,
        filename=f"{output_dir}/value_function_epoch_{epoch}.pdf"
    )

    # Plot generator with GV discretization
    visualize_generator(
        V_net=V_net,
        GV_net=GV_net,
        regions=regions,
        epoch=epoch,
        show_regions=True,
        show_discretization=True,
        training_cells=gv_cells,
        filename=f"{output_dir}/generator_epoch_{epoch}.pdf"
    )

    print(f"Saved visualization plots for epoch {epoch} to '{output_dir}'")

def plot_constraint_regions(
    V_net,
    regions: Regions,
    beta_s: float,
    beta_ra: float,
    filename: Optional[str] = None,
    resolution: int = 100,
    figsize: Tuple[int, int] = (10, 8)
):
    V_net.eval()

    full_bounds = regions.full.bounds
    D = full_bounds.shape[0]
    pairs = _get_plot_pairs(D)

    try:
        net_device = next(V_net.parameters()).device
    except StopIteration:
        net_device = torch.device("cpu")

    n_plots = len(pairs)
    fig, axes = plt.subplots(
        1, n_plots,
        figsize=(figsize[0] * n_plots, figsize[1]),
        squeeze=False
    )
    axes = axes.ravel()

    for p, (x_dim, y_dim) in enumerate(pairs):
        ax = axes[p]

        slice_point = make_slice_point_for_region(
            regions.full,   # or regions.goal / regions.init / regions.full
            full_bounds,
            x_dim=x_dim,
            y_dim=y_dim,
        )

        X, Y, grid_np = _make_slice_grid(full_bounds, x_dim, y_dim, resolution, slice_point=slice_point)
        x_grid = torch.tensor(grid_np, dtype=torch.float32, device=net_device)

        with torch.no_grad():
            V_grid = V_net(x_grid).detach().cpu().numpy().reshape(X.shape)

        contour = ax.contourf(X, Y, V_grid, levels=20, cmap='viridis')
        fig.colorbar(contour, ax=ax, label=f"V({_format_dim_label(x_dim)}, {_format_dim_label(y_dim)})")

        # constraint contours
        ax.contour(X, Y, V_grid, levels=[beta_s / 2.0], colors='cyan', linewidths=2, linestyles='--')
        ax.contour(X, Y, V_grid, levels=[beta_s], colors='yellow', linewidths=2, linestyles='--')
        ax.contour(X, Y, V_grid, levels=[1.0], colors='green', linewidths=2, linestyles='--')
        ax.contour(X, Y, V_grid, levels=[beta_ra], colors='orange', linewidths=2, linestyles='--')

        ax.set_xlabel(_format_dim_label(x_dim), fontsize=12)
        ax.set_ylabel(_format_dim_label(y_dim), fontsize=12)
        ax.set_title(f"{_format_dim_label(x_dim)} vs {_format_dim_label(y_dim)}",
                     fontsize=12, fontweight='bold')

        # regions projected
        lab_init  = "Init" if p == 0 else None
        lab_unsafe = "Unsafe" if p == 0 else None
        lab_goal  = "Goal" if p == 0 else None
        _draw_region_proj(ax, regions.init,   'green', lab_init,  x_dim, y_dim)
        _draw_region_proj(ax, regions.unsafe, 'red',   lab_unsafe, x_dim, y_dim)
        _draw_region_proj(ax, regions.goal,   'blue',  lab_goal,  x_dim, y_dim)

        if p == 0:
            ax.legend(loc='upper right', fontsize=9)

    fig.suptitle("Value Function with Constraint Boundaries", fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if filename is None:
        filename = "constraint_regions.pdf"
    fig.savefig(filename, dpi = 500, format='pdf')
    plt.close(fig)

    V_net.train()


def plot_loss_history(
    loss_history: List[dict],
    refinement_epochs: Optional[dict] = None,
    filename: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 7)
):
    """
    Plot training loss history with stacked area chart showing contribution of each component.

    Args:
        loss_history: List of loss dictionaries from training
        refinement_epochs: Dict with 'outside' and 'generator' keys containing lists of refinement epochs (optional)
        filename: Output filename
        figsize: Figure size
    """
    if len(loss_history) == 0:
        print("No loss history to plot")
        return

    epochs = np.array([d['epoch'] for d in loss_history])

    # Extract loss components
    init_loss = np.array([d['init'] for d in loss_history])
    unsafe_loss = np.array([d['unsafe'] for d in loss_history])
    outside_loss = np.array([d['outside'] for d in loss_history])
    generator_loss = np.array([d['generator'] for d in loss_history])
    total_loss = np.array([d['total'] for d in loss_history])

    # Create single figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.suptitle("Training Loss History - Component Contributions", fontsize=16, fontweight='bold')

    # Stacked area chart showing contribution of each loss component
    ax.stackplot(epochs, init_loss, unsafe_loss, outside_loss, generator_loss,
                 labels=['Init', 'Unsafe', 'Outside', 'Generator'],
                 colors=['#2ecc71', '#e74c3c', 'orange', '#9b59b6'],
                 alpha=0.8)

    # Overlay total loss as a thick black line
    ax.plot(epochs, total_loss, 'k-', linewidth=3, label='Total Loss', zorder=10)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss Magnitude', fontsize=12)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, zorder=0)

    # Add refinement epoch markers with improved visualization
    refinement_legend_items = []

    if refinement_epochs is not None:
        # Outside refinements - use solid vertical spans
        outside_refs = refinement_epochs.get('outside', [])
        if len(outside_refs) > 0:
            for i, ref_epoch in enumerate(outside_refs):
                # Use narrow vertical spans instead of lines for better visibility
                ax.axvspan(ref_epoch - 5, ref_epoch + 5, color='orange', alpha=0.15, zorder=0)
                line = ax.axvline(x=ref_epoch, color='darkorange', linestyle='-',
                                 linewidth=2.5, alpha=0.85, zorder=1)
                if i == 0:
                    refinement_legend_items.append((line, 'Outside Refine'))

        # Generator refinements - use different color
        generator_refs = refinement_epochs.get('generator', [])
        if len(generator_refs) > 0:
            for i, ref_epoch in enumerate(generator_refs):
                # Use narrow vertical spans
                ax.axvspan(ref_epoch - 5, ref_epoch + 5, color='purple', alpha=0.12, zorder=0)
                line = ax.axvline(x=ref_epoch, color='purple', linestyle='-',
                                 linewidth=2.5, alpha=0.85, zorder=1)
                if i == 0:
                    refinement_legend_items.append((line, 'Generator Refine'))

        # Add refinement markers to the legend
        if refinement_legend_items:
            # Get existing legend items
            handles, labels = ax.get_legend_handles_labels()
            # Add refinement markers
            for line, label in refinement_legend_items:
                handles.append(line)
                labels.append(label)
            ax.legend(handles, labels, loc='upper right', fontsize=11, framealpha=0.95)

    plt.tight_layout()

    # Save
    if filename is None:
        filename = "loss_history.pdf"
    plt.savefig(filename, dpi=500, format='pdf')
    plt.close()


def create_summary_plots(
    V_net,
    GV_net,
    regions: Regions,
    region_cells: dict,
    beta_s: float,
    beta_ra: float,
    loss_history: Optional[List[dict]] = None,
    refinement_epochs: Optional[dict] = None,
    results=None,
    output_dir: str = "results"
):
    """
    Create a comprehensive set of summary plots.

    Args:
        V_net: Value function network
        GV_net: Generator network
        regions: Regions object
        region_cells: Dictionary of region cells
        beta_s: Separation threshold
        beta_ra: Unsafe threshold
        loss_history: Training loss history (optional)
        refinement_epochs: Dict with 'outside' and 'generator' refinement epochs (optional)
        output_dir: Output directory for plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Cells for V network (all regions except generator)
    v_cells = []
    for region_name in ['init', 'goal', 'unsafe', 'outside']:
        if region_name in region_cells:
            v_cells.extend(region_cells[region_name])

    # Cells for generator network (only generator region)
    gv_cells = region_cells.get('generator', [])

    # Value function with V discretization
    visualize_value_function(
        V_net, regions,
        show_regions=True,
        show_discretization=True,
        training_cells=v_cells,
        filename=f"{output_dir}/value_function.pdf"
    )

    # Generator with GV discretization
    visualize_generator(
        V_net, GV_net, regions,
        show_regions=True,
        show_discretization=True,
        training_cells=gv_cells,
        results=results,
        filename=f"{output_dir}/generator.pdf"
    )

    # Constraint regions
    plot_constraint_regions(
        V_net, regions, beta_s, beta_ra,
        filename=f"{output_dir}/constraint_regions.pdf"
    )

    # Loss history
    if loss_history is not None and len(loss_history) > 0:
        plot_loss_history(
            loss_history,
            refinement_epochs=refinement_epochs,
            filename=f"{output_dir}/loss_history.pdf"
        )

    print(f"All plots saved to '{output_dir}/'")
