"""
Visualization utilities for plotting value functions and generators.

This module provides functions to:
- Plot value function V(x) over the state space
- Plot generator Φ(x) over the state space
- Overlay regions (init, goal, unsafe)
- Show discretization cells
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Optional, List, Tuple

from regions import Regions


def visualize_value_function(
    V_net,
    regions: Regions,
    title: str = "Value Function V(x)",
    show_regions: bool = True,
    show_discretization: bool = False,
    training_cells: Optional[List] = None,
    filename: Optional[str] = None,
    resolution: int = 100,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Visualize the value function V(x) as a contour plot.

    Args:
        V_net: Value function network
        regions: Regions object with init, goal, unsafe, full
        title: Plot title
        show_regions: Whether to overlay region boundaries
        show_discretization: Whether to show discretization cells
        training_cells: List of (lower, upper) cell tuples (optional)
        filename: Output filename (default: "value_function.png")
        resolution: Grid resolution for plotting
        figsize: Figure size
    """
    V_net.eval()

    # Create grid over full region
    full_bounds = regions.full.bounds
    x1_vals = np.linspace(full_bounds[0, 0], full_bounds[0, 1], resolution)
    x2_vals = np.linspace(full_bounds[1, 0], full_bounds[1, 1], resolution)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)

    # Flatten and evaluate
    x1_flat = X1.flatten()
    x2_flat = X2.flatten()
    x_grid = torch.tensor(np.stack([x1_flat, x2_flat], axis=1), dtype=torch.float32)

    with torch.no_grad():
        V_output = V_net(x_grid).numpy().flatten()

    # Reshape for contour plot
    V_grid = V_output.reshape(X1.shape)

    # Get output range
    vmin, vmax = V_grid.min(), V_grid.max()

    # Create plot
    plt.figure(figsize=figsize)
    contour = plt.contourf(X1, X2, V_grid, levels=20, cmap='viridis')
    plt.colorbar(contour, label='V(x₁, x₂)')

    plt.xlabel('x₁', fontsize=12)
    plt.ylabel('x₂', fontsize=12)
    plt.title(f"{title}\nRange: [{vmin:.4f}, {vmax:.4f}]", fontsize=14, fontweight='bold')

    # Overlay regions
    if show_regions:
        _draw_region(regions.init, 'green', 'Init')
        _draw_region(regions.unsafe, 'red', 'Unsafe')
        _draw_region(regions.goal, 'blue', 'Goal')
        plt.legend(loc='upper right', fontsize=10)

    # Draw discretization cells
    if show_discretization and training_cells is not None:
        for cell_lower, cell_upper in training_cells:
            cell_rect = Rectangle(
                (cell_lower[0].item(), cell_lower[1].item()),
                cell_upper[0].item() - cell_lower[0].item(),
                cell_upper[1].item() - cell_lower[1].item(),
                linewidth=0.5, edgecolor='black', facecolor='none', alpha=0.5
            )
            plt.gca().add_patch(cell_rect)

    plt.tight_layout()

    # Save
    if filename is None:
        filename = "value_function.png"
    plt.savefig(filename, dpi=150)
    print(f"  → Saved to '{filename}'")
    plt.close()

    V_net.train()


def visualize_generator(
    V_net,
    GV_net,
    regions: Regions,
    title: str = "Generator Φ(x)",
    show_regions: bool = True,
    show_discretization: bool = False,
    training_cells: Optional[List] = None,
    filename: Optional[str] = None,
    resolution: int = 100,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Visualize the generator Φ(x) = f·∇V + 0.5·Tr(g·g^T·H_V) as a contour plot.

    Args:
        V_net: Value function network
        GV_net: Generator network
        regions: Regions object with init, goal, unsafe, full
        title: Plot title
        show_regions: Whether to overlay region boundaries
        show_discretization: Whether to show discretization cells
        training_cells: List of (lower, upper) cell tuples (optional)
        filename: Output filename (default: "generator.png")
        resolution: Grid resolution for plotting
        figsize: Figure size
    """
    V_net.eval()
    GV_net.eval()

    # Create grid over full region
    full_bounds = regions.full.bounds
    x1_vals = np.linspace(full_bounds[0, 0], full_bounds[0, 1], resolution)
    x2_vals = np.linspace(full_bounds[1, 0], full_bounds[1, 1], resolution)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)

    # Flatten and evaluate
    x1_flat = X1.flatten()
    x2_flat = X2.flatten()
    x_grid = torch.tensor(np.stack([x1_flat, x2_flat], axis=1), dtype=torch.float32)

    with torch.no_grad():
        Phi_output = GV_net(x_grid).numpy().flatten()

    # Reshape for contour plot
    Phi_grid = Phi_output.reshape(X1.shape)

    # Get output range
    phi_min, phi_max = Phi_grid.min(), Phi_grid.max()

    # Create plot - use diverging colormap centered at 0
    plt.figure(figsize=figsize)

    # Symmetric colormap around 0
    abs_max = max(abs(phi_min), abs(phi_max))
    contour = plt.contourf(X1, X2, Phi_grid, levels=20, cmap='RdBu_r',
                           vmin=-abs_max, vmax=abs_max)
    plt.colorbar(contour, label='Φ(x₁, x₂)')

    # Add contour line at Φ=0
    plt.contour(X1, X2, Phi_grid, levels=[0], colors='black', linewidths=2)

    plt.xlabel('x₁', fontsize=12)
    plt.ylabel('x₂', fontsize=12)
    plt.title(f"{title}\nRange: [{phi_min:.4f}, {phi_max:.4f}]", fontsize=14, fontweight='bold')

    # Overlay regions
    if show_regions:
        _draw_region(regions.init, 'green', 'Init')
        _draw_region(regions.unsafe, 'red', 'Unsafe')
        _draw_region(regions.goal, 'blue', 'Goal')
        plt.legend(loc='upper right', fontsize=10)

    # Draw discretization cells
    if show_discretization and training_cells is not None:
        for cell_lower, cell_upper in training_cells:
            cell_rect = Rectangle(
                (cell_lower[0].item(), cell_lower[1].item()),
                cell_upper[0].item() - cell_lower[0].item(),
                cell_upper[1].item() - cell_lower[1].item(),
                linewidth=0.5, edgecolor='black', facecolor='none', alpha=0.5
            )
            plt.gca().add_patch(cell_rect)

    plt.tight_layout()

    # Save
    if filename is None:
        filename = "generator.png"
    plt.savefig(filename, dpi=150)
    print(f"  → Saved to '{filename}'")
    plt.close()

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
    Visualize training progress (both V and Φ) with discretization.

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
        title=f"Value Function V(x) - Epoch {epoch}",
        show_regions=True,
        show_discretization=True,
        training_cells=v_cells,
        filename=f"{output_dir}/value_function_epoch_{epoch}.png"
    )

    # Plot generator with GV discretization
    visualize_generator(
        V_net=V_net,
        GV_net=GV_net,
        regions=regions,
        title=f"Generator Φ(x) - Epoch {epoch}",
        show_regions=True,
        show_discretization=True,
        training_cells=gv_cells,
        filename=f"{output_dir}/generator_epoch_{epoch}.png"
    )


def plot_constraint_regions(
    V_net,
    regions: Regions,
    beta_s: float,
    beta_ra: float,
    filename: Optional[str] = None,
    resolution: int = 100,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Plot value function with constraint boundaries marked.

    Args:
        V_net: Value function network
        regions: Regions object
        beta_s: Separation threshold
        beta_ra: Unsafe threshold
        filename: Output filename
        resolution: Grid resolution
        figsize: Figure size
    """
    V_net.eval()

    # Create grid
    full_bounds = regions.full.bounds
    x1_vals = np.linspace(full_bounds[0, 0], full_bounds[0, 1], resolution)
    x2_vals = np.linspace(full_bounds[1, 0], full_bounds[1, 1], resolution)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)

    # Evaluate
    x1_flat = X1.flatten()
    x2_flat = X2.flatten()
    x_grid = torch.tensor(np.stack([x1_flat, x2_flat], axis=1), dtype=torch.float32)

    with torch.no_grad():
        V_output = V_net(x_grid).numpy().flatten()

    V_grid = V_output.reshape(X1.shape)

    # Create plot
    plt.figure(figsize=figsize)
    contour = plt.contourf(X1, X2, V_grid, levels=20, cmap='viridis')
    plt.colorbar(contour, label='V(x₁, x₂)')

    # Add constraint boundaries
    plt.contour(X1, X2, V_grid, levels=[beta_s/2.0], colors='cyan',
               linewidths=2, linestyles='--', label=f'V={beta_s/2.0} (goal target)')
    plt.contour(X1, X2, V_grid, levels=[beta_s], colors='yellow',
               linewidths=2, linestyles='--', label=f'V={beta_s} (separation)')
    plt.contour(X1, X2, V_grid, levels=[beta_ra], colors='orange',
               linewidths=2, linestyles='--', label=f'V={beta_ra} (unsafe)')

    plt.xlabel('x₁', fontsize=12)
    plt.ylabel('x₂', fontsize=12)
    plt.title("Value Function with Constraint Boundaries", fontsize=14, fontweight='bold')

    # Overlay regions
    _draw_region(regions.init, 'green', 'Init')
    _draw_region(regions.unsafe, 'red', 'Unsafe')
    _draw_region(regions.goal, 'blue', 'Goal')

    plt.legend(loc='upper right', fontsize=9)
    plt.tight_layout()

    # Save
    if filename is None:
        filename = "constraint_regions.png"
    plt.savefig(filename, dpi=150)
    print(f"  → Saved to '{filename}'")
    plt.close()

    V_net.train()


def plot_loss_history(
    loss_history: List[dict],
    filename: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
):
    """
    Plot training loss history.

    Args:
        loss_history: List of loss dictionaries from training
        filename: Output filename
        figsize: Figure size
    """
    if len(loss_history) == 0:
        print("No loss history to plot")
        return

    epochs = [d['epoch'] for d in loss_history]

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle("Training Loss History", fontsize=16, fontweight='bold')

    # Total loss
    axes[0, 0].plot(epochs, [d['total'] for d in loss_history], 'k-', linewidth=2)
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)

    # Goal loss
    axes[0, 1].plot(epochs, [d['goal'] for d in loss_history], 'b-', linewidth=2)
    axes[0, 1].set_title('Goal Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)

    # Unsafe loss
    axes[0, 2].plot(epochs, [d['unsafe'] for d in loss_history], 'r-', linewidth=2)
    axes[0, 2].set_title('Unsafe Loss')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].grid(True, alpha=0.3)

    # Init loss
    axes[1, 0].plot(epochs, [d['init'] for d in loss_history], 'g-', linewidth=2)
    axes[1, 0].set_title('Init Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True, alpha=0.3)

    # Outside loss
    axes[1, 1].plot(epochs, [d['outside'] for d in loss_history], 'c-', linewidth=2)
    axes[1, 1].set_title('Outside Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].grid(True, alpha=0.3)

    # Generator loss
    axes[1, 2].plot(epochs, [d['generator'] for d in loss_history], 'm-', linewidth=2)
    axes[1, 2].set_title('Generator Loss')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Loss')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    if filename is None:
        filename = "loss_history.png"
    plt.savefig(filename, dpi=150)
    print(f"  → Saved to '{filename}'")
    plt.close()


def _draw_region(region, color: str, label: str):
    """Helper to draw a region rectangle."""
    bounds = region.bounds
    rect = Rectangle(
        (bounds[0, 0], bounds[1, 0]),
        bounds[0, 1] - bounds[0, 0],
        bounds[1, 1] - bounds[1, 0],
        linewidth=3, edgecolor=color, facecolor='none', label=label
    )
    plt.gca().add_patch(rect)


def create_summary_plots(
    V_net,
    GV_net,
    regions: Regions,
    region_cells: dict,
    beta_s: float,
    beta_ra: float,
    loss_history: Optional[List[dict]] = None,
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
        output_dir: Output directory for plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("CREATING SUMMARY PLOTS")
    print("="*80)

    # Cells for V network (all regions except generator)
    v_cells = []
    for region_name in ['init', 'goal', 'unsafe', 'outside']:
        if region_name in region_cells:
            v_cells.extend(region_cells[region_name])

    # Cells for generator network (only generator region)
    gv_cells = region_cells.get('generator', [])

    # Value function with V discretization
    print("\n1. Value function with discretization...")
    visualize_value_function(
        V_net, regions,
        title="Value Function V(x)",
        show_regions=True,
        show_discretization=True,
        training_cells=v_cells,
        filename=f"{output_dir}/value_function.png"
    )

    # Generator with GV discretization
    print("\n2. Generator with discretization...")
    visualize_generator(
        V_net, GV_net, regions,
        title="Generator Φ(x)",
        show_regions=True,
        show_discretization=True,
        training_cells=gv_cells,
        filename=f"{output_dir}/generator.png"
    )

    # Constraint regions
    print("\n3. Constraint regions...")
    plot_constraint_regions(
        V_net, regions, beta_s, beta_ra,
        filename=f"{output_dir}/constraint_regions.png"
    )

    # Loss history
    if loss_history is not None and len(loss_history) > 0:
        print("\n4. Loss history...")
        plot_loss_history(
            loss_history,
            filename=f"{output_dir}/loss_history.png"
        )

    print(f"\nAll plots saved to '{output_dir}/'")
