"""
Minimal, publication-ready animation for 3D Nonlinear System.
Layout: 3D state plot (left) + 3 control plots (right).
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def animate_nonlinear3d_minimal(
    *,
    f_cl_module,
    f_open_module=None,
    g_fn=None,
    init_range: np.ndarray,
    goal_range: np.ndarray,
    full_range: np.ndarray,
    unsafe_boxes: np.ndarray,
    V_net=None,
    device: str = "cpu",
    dt: float = 0.01,
    T: float = 10.0,
    seed: int = 0,
    save_path: str | None = None,
    save_final_frame: str | None = None,
    show: bool = True,
    controller_label: str | None = None,
    n_trajectories: int = 5,
):
    """
    Minimal, publication-ready animation with only essential plots.
    Focuses on: (1) 3D state trajectory, (2) Control plots

    Args:
        f_cl_module: Closed-loop dynamics module
        f_open_module: Optional open-loop (non-trained) dynamics module
        g_fn: Optional diffusion function
        init_range: Initial region bounds (3, 2)
        goal_range: Goal region bounds (3, 2)
        full_range: Full state space bounds (3, 2)
        unsafe_boxes: Unsafe region boxes
        device: Torch device
        dt: Time step
        T: Total time
        seed: Random seed
        save_path: Path to save animation video (MP4)
        save_final_frame: Path to save final frame as high-quality image (PDF/PNG)
        show: Whether to display animation
        controller_label: Label for controller
        n_trajectories: Number of stochastic trajectories to simulate
    """
    rng = np.random.default_rng(seed)

    init_range = np.asarray(init_range, dtype=np.float32)
    goal_range = np.asarray(goal_range, dtype=np.float32)
    full_range = np.asarray(full_range, dtype=np.float32)

    def _as_union_boxes(x_unsafe: np.ndarray) -> np.ndarray:
        x = np.asarray(x_unsafe, dtype=np.float32)
        if x.ndim == 2:
            if x.shape == (3, 2):
                return x[None, ...]
            if x.shape[1] == 2 and (x.shape[0] % 3 == 0):
                K = x.shape[0] // 3
                return x.reshape(K, 3, 2)
            raise ValueError(f"unsafe_boxes 2D must be (3,2) or (K*3,2), got {x.shape}")
        if x.ndim == 3:
            if x.shape[1:] != (3, 2):
                raise ValueError(f"unsafe_boxes 3D must be (K,3,2), got {x.shape}")
            return x
        raise ValueError(f"unsafe_boxes must be (3,2), (K*3,2), or (K,3,2), got {x.shape}")

    unsafeK = _as_union_boxes(unsafe_boxes)

    # Helper to check if state is in goal region
    def in_goal(x_state):
        return np.all(x_state >= goal_range[:, 0]) and np.all(x_state <= goal_range[:, 1])

    # Store initial conditions for reuse in open-loop simulation
    initial_conditions = []

    # Simulate multiple trajectories with different initial conditions
    N_max = int(T / dt) + 1
    all_trajectories_closed = []
    all_controls_closed = []
    all_times_closed = []

    all_trajectories_open = []
    all_controls_open = []
    all_times_open = []

    f_cl_module.eval()
    if f_open_module is not None:
        f_open_module.eval()

    print(f"Simulating {n_trajectories} trajectories...")
    for traj_idx in range(n_trajectories):
        # Sample initial condition
        x0 = np.array([
            rng.uniform(init_range[0, 0], init_range[0, 1]),
            rng.uniform(init_range[1, 0], init_range[1, 1]),
            rng.uniform(init_range[2, 0], init_range[2, 1]),
        ], dtype=np.float32)
        initial_conditions.append(x0)

        # Simulate closed-loop (controlled) trajectory
        X = np.zeros((N_max, 3), dtype=np.float32)
        U = np.zeros((N_max, 3), dtype=np.float32)
        X[0] = x0

        N = N_max
        goal_reached_step = None
        extra_steps_after_goal = int(2.0 / dt)  # 2 seconds worth of steps

        with torch.no_grad():
            for k in range(N_max - 1):
                xk_t = torch.tensor(X[k:k + 1], dtype=torch.float32, device=device)

                # Get control from controller if available
                if hasattr(f_cl_module, 'controller'):
                    uk = f_cl_module.controller(xk_t).detach().cpu().numpy().reshape(3)
                else:
                    uk = np.zeros(3, dtype=np.float32)

                # Get closed-loop drift (f + u)
                xdot = f_cl_module(xk_t).detach().cpu().numpy().reshape(3)

                if g_fn is None:
                    xnext = X[k] + dt * xdot
                else:
                    gk = g_fn(xk_t).detach().cpu().numpy().reshape(3)
                    dW = (np.sqrt(dt) * rng.standard_normal(3)).astype(np.float32)
                    xnext = X[k] + dt * xdot + gk * dW

                X[k + 1] = xnext
                U[k] = uk

                # Track when goal is first reached
                if goal_reached_step is None and in_goal(xnext):
                    goal_reached_step = k + 1
                    print(f"Closed-loop Trajectory {traj_idx+1}: Goal reached at t={(k+1)*dt:.2f}s")

                # Stop simulation 2 seconds after reaching goal
                if goal_reached_step is not None and (k + 1 - goal_reached_step) >= extra_steps_after_goal:
                    N = k + 2
                    X = X[:N]
                    U = U[:N]
                    print(f"Closed-loop Trajectory {traj_idx+1}: Stopped at t={(N-1)*dt:.2f}s (2s after goal)")
                    break

            U[-1] = U[-2]

        t_traj = np.linspace(0.0, (N-1)*dt, N, dtype=np.float32)

        all_trajectories_closed.append(X)
        all_controls_closed.append(U)
        all_times_closed.append(t_traj)

    # Find the maximum time across all controlled trajectories
    max_closed_time = max(t_traj[-1] for t_traj in all_times_closed)
    max_closed_steps = int(max_closed_time / dt) + 1
    print(f"Maximum controlled trajectory time: {max_closed_time:.2f}s ({max_closed_steps} steps)")

    # Simulate open-loop trajectories if module provided
    if f_open_module is not None:
        print(f"Simulating {n_trajectories} open-loop trajectories...")
        for traj_idx in range(n_trajectories):
            x0 = initial_conditions[traj_idx]

            X_open = np.zeros((max_closed_steps, 3), dtype=np.float32)
            U_open = np.zeros((max_closed_steps, 3), dtype=np.float32)
            X_open[0] = x0

            with torch.no_grad():
                for k in range(max_closed_steps - 1):
                    xk_t = torch.tensor(X_open[k:k + 1], dtype=torch.float32, device=device)
                    xdot_open = f_open_module(xk_t).detach().cpu().numpy().reshape(3)

                    if g_fn is None:
                        xnext_open = X_open[k] + dt * xdot_open
                    else:
                        gk = g_fn(xk_t).detach().cpu().numpy().reshape(3)
                        dW = (np.sqrt(dt) * rng.standard_normal(3)).astype(np.float32)
                        xnext_open = X_open[k] + dt * xdot_open + gk * dW

                    X_open[k + 1] = xnext_open
                    # Open-loop has zero control
                    U_open[k] = np.zeros(3, dtype=np.float32)

                U_open[-1] = U_open[-2]

            t_traj_open = np.linspace(0.0, (max_closed_steps-1)*dt, max_closed_steps, dtype=np.float32)

            all_trajectories_open.append(X_open)
            all_controls_open.append(U_open)
            all_times_open.append(t_traj_open)

    # Use the longest closed-loop trajectory for animation timing
    max_len = max(len(traj) for traj in all_trajectories_closed)
    t = np.linspace(0.0, (max_len-1)*dt, max_len, dtype=np.float32)
    N = max_len

    # Clean styling with Times New Roman
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'axes.linewidth': 0.8,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18,
        'legend.framealpha': 0.9,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
    })

    # Colors matching src/visualization.py
    colors = {
        'init': 'seagreen',
        'goal': 'darkgoldenrod',
        'unsafe': 'firebrick',
        'full': '#95a5a6',
    }

    # Colors: use distinct colormaps that are visible against dark background
    cmap_open = plt.cm.binary  # Binary shades for uncontrolled
    cmap_closed = plt.cm.inferno  # Inferno shades for controlled

    # Generate colors from colormaps (use brighter end of spectrum)
    # colors_open = [cmap_open(0.1 + 0.1 * i) for i in range(n_trajectories)]
    # colors_closed = [cmap_closed(0.1 + 0.1 * i) for i in range(n_trajectories)]
    colors_open = ["black"] * n_trajectories
    colors_closed = ["deeppink"] * n_trajectories

    # Layout: Single row with 4 plots
    # From left to right: Controlled 3D, Uncontrolled 3D, x1 vs x2 phase plane, x2 vs x3 phase plane
    fig = plt.figure(figsize=(28, 5.5), dpi=120)

    if controller_label:
        fig.suptitle(controller_label, fontsize=18, y=0.98, weight='medium')

    # Create two grid specs - one for 3D plots (tight spacing), one for phase planes (normal spacing)
    # Left section: two 3D plots with tight spacing
    gs_3d = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0],
                             wspace=0.00,
                             top=0.94, bottom=0.08, left=0.1, right=0.48)

    # Right section: two phase plane plots with normal spacing
    gs_phase = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0],
                                wspace=0.23,
                                top=0.94, bottom=0.08, left=0.53, right=0.98)

    ax_3d_closed = fig.add_subplot(gs_3d[0, 0], projection="3d")  # Controlled 3D - leftmost
    ax_3d_open = fig.add_subplot(gs_3d[0, 1], projection="3d")  # Uncontrolled 3D - second from left
    ax_phase1 = fig.add_subplot(gs_phase[0, 0])  # x1 vs x2 - third from left
    ax_phase2 = fig.add_subplot(gs_phase[0, 1])  # x2 vs x3 - rightmost

    # ========== LEFT: 3D State Trajectories ==========
    def _draw_box3d(ax3d, box3x2, *, lw=1.2, color="k", alpha=0.6, label=None):
        """Draw a 3D box with proper legend support."""
        x0, x1 = float(box3x2[0, 0]), float(box3x2[0, 1])
        y0, y1 = float(box3x2[1, 0]), float(box3x2[1, 1])
        z0, z1 = float(box3x2[2, 0]), float(box3x2[2, 1])

        # 8 corners
        p000 = (x0, y0, z0)
        p001 = (x0, y0, z1)
        p010 = (x0, y1, z0)
        p011 = (x0, y1, z1)
        p100 = (x1, y0, z0)
        p101 = (x1, y0, z1)
        p110 = (x1, y1, z0)
        p111 = (x1, y1, z1)

        faces = [
            [p000, p001, p011, p010],  # x=x0
            [p100, p101, p111, p110],  # x=x1
            [p000, p001, p101, p100],  # y=y0
            [p010, p011, p111, p110],  # y=y1
            [p000, p010, p110, p100],  # z=z0
            [p001, p011, p111, p101],  # z=z1
        ]

        poly = Poly3DCollection(faces, alpha=alpha, linewidths=lw)
        poly.set_facecolor(color)
        poly.set_edgecolor(color)
        ax3d.add_collection3d(poly)

        # Add to legend by creating a dummy plot with the same color
        if label is not None:
            ax3d.plot([], [], [], color=color, linewidth=lw, label=label)

        return poly

    # ========== TOP LEFT: 3D Controlled ==========
    ax_3d_closed.set_title('Controlled', fontsize=25, weight='medium', pad=2)

    # Draw regions (NO unsafe regions for zoomed-in view)
    _draw_box3d(ax_3d_closed, init_range, lw=1.5, color=colors['init'], alpha=0.1, label='Init')
    _draw_box3d(ax_3d_closed, goal_range, lw=1.5, color=colors['goal'], alpha=0.1, label='Goal')

    # ========== TOP RIGHT: 3D Uncontrolled ==========
    ax_3d_open.set_title('Uncontrolled', fontsize=25, weight='medium', pad=2)

    # Draw regions
    _draw_box3d(ax_3d_open, init_range, lw=1.5, color=colors['init'], alpha=0.1, label='Init')
    _draw_box3d(ax_3d_open, goal_range, lw=1.5, color=colors['goal'], alpha=0.1, label='Goal')
    for k in range(unsafeK.shape[0]):
        _draw_box3d(ax_3d_open, unsafeK[k], lw=1.2, color=colors['unsafe'], alpha=0.05,
                   label='Unsafe' if k == 0 else None)

    # Trajectory lines and points for controlled
    lines3d_closed = []
    pts3d_closed = []
    for i in range(n_trajectories):
        line_closed, = ax_3d_closed.plot([], [], [], lw=1.5, color=colors_closed[i], alpha=0.8)
        pt_closed, = ax_3d_closed.plot([], [], [], marker='o', markersize=4, color=colors_closed[i])
        lines3d_closed.append(line_closed)
        pts3d_closed.append(pt_closed)

    # Trajectory lines and points for uncontrolled
    lines3d_open = []
    pts3d_open = []
    for i in range(n_trajectories):
        line_open, = ax_3d_open.plot([], [], [], lw=1.5, color=colors_open[i], alpha=0.8)
        pt_open, = ax_3d_open.plot([], [], [], marker='s', markersize=4, color=colors_open[i])
        lines3d_open.append(line_open)
        pts3d_open.append(pt_open)

    # Styling for controlled 3D - zoomed in to see controlled trajectories clearly
    ax_3d_closed.set_xlabel('$x_1$', labelpad=6, fontsize=25)
    ax_3d_closed.set_ylabel('$x_2$', labelpad=6, fontsize=25)
    ax_3d_closed.set_zlabel('$x_3$', labelpad=6, fontsize=25)
    # Zoom in by shrinking the view
    zoom_in_factor = 0.12
    for dim, ax_method in enumerate([ax_3d_closed.set_xlim, ax_3d_closed.set_ylim, ax_3d_closed.set_zlim]):
        # center = 0.5 * (full_range[dim, 0] + full_range[dim, 1])
        center = 0
        span = (full_range[dim, 1] - full_range[dim, 0]) * zoom_in_factor
        ax_method(center - span/2, center + span/2)
    ax_3d_closed.view_init(elev=15, azim=135)
    ax_3d_closed.legend(loc='upper left', framealpha=0.9, fontsize=12)
    ax_3d_closed.grid(True, alpha=0.2)

    # Styling for uncontrolled 3D - zoomed out to see full divergence
    ax_3d_open.set_xlabel('$x_1$', labelpad=6, fontsize=25)
    ax_3d_open.set_ylabel('$x_2$', labelpad=6, fontsize=25)
    ax_3d_open.set_zlabel('$x_3$', labelpad=6, fontsize=25)
    # Zoom out by expanding the view (e.g., 1.5x of full_range)
    zoom_out_factor = 5.5
    for dim, ax_method in enumerate([ax_3d_open.set_xlim, ax_3d_open.set_ylim, ax_3d_open.set_zlim]):
        center = 0.5 * (full_range[dim, 0] + full_range[dim, 1])
        span = (full_range[dim, 1] - full_range[dim, 0]) * zoom_out_factor
        ax_method(center - span/2, center + span/2)
    ax_3d_open.view_init(elev=15, azim=135)
    ax_3d_open.legend(loc='upper left', framealpha=0.9, fontsize=12)
    ax_3d_open.grid(True, alpha=0.2)

    # ========== BOTTOM: Two phase plane plots ==========
    from matplotlib.patches import Rectangle

    # Precompute V(x) contour data if V_net provided
    V_contour_data_12 = None
    V_contour_data_23 = None
    if V_net is not None:
        V_net.eval()
        resolution = 50

        # For x1 vs x2 plane (fix x3 at center)
        x1_grid = np.linspace(full_range[0, 0], full_range[0, 1], resolution)
        x2_grid = np.linspace(full_range[1, 0], full_range[1, 1], resolution)
        X1_mesh, X2_mesh = np.meshgrid(x1_grid, x2_grid)
        x3_center = 0.5 * (full_range[2, 0] + full_range[2, 1])

        grid_pts_12 = np.stack([
            X1_mesh.ravel(),
            X2_mesh.ravel(),
            np.full(X1_mesh.size, x3_center)
        ], axis=1)

        with torch.no_grad():
            grid_tensor_12 = torch.tensor(grid_pts_12, dtype=torch.float32, device=device)
            V_vals_12 = V_net(grid_tensor_12).detach().cpu().numpy().reshape(X1_mesh.shape)

        V_contour_data_12 = (X1_mesh, X2_mesh, V_vals_12)

        # For x2 vs x3 plane (fix x1 at center)
        x2_grid = np.linspace(full_range[1, 0], full_range[1, 1], resolution)
        x3_grid = np.linspace(full_range[2, 0], full_range[2, 1], resolution)
        X2_mesh, X3_mesh = np.meshgrid(x2_grid, x3_grid)
        x1_center = 0.5 * (full_range[0, 0] + full_range[0, 1])

        grid_pts_23 = np.stack([
            np.full(X2_mesh.size, x1_center),
            X2_mesh.ravel(),
            X3_mesh.ravel()
        ], axis=1)

        with torch.no_grad():
            grid_tensor_23 = torch.tensor(grid_pts_23, dtype=torch.float32, device=device)
            V_vals_23 = V_net(grid_tensor_23).detach().cpu().numpy().reshape(X2_mesh.shape)

        V_contour_data_23 = (X2_mesh, X3_mesh, V_vals_23)

    # Phase plane 1: x1 vs x2
    ax_phase1.set_xlabel('$x_1$', fontsize=25)
    ax_phase1.set_ylabel('$x_2$', fontsize=25)
    # ax_phase1.set_title('$x_1$ vs $x_2$', fontsize=25, weight='medium')
    ax_phase1.grid(True, alpha=0.3)
    ax_phase1.set_xlim(float(full_range[0, 0]), float(full_range[0, 1]))
    ax_phase1.set_ylim(float(full_range[1, 0]), float(full_range[1, 1]))

    # Draw V contour if available
    if V_contour_data_12 is not None:
        X1_mesh, X2_mesh, V_vals_12 = V_contour_data_12
        contour = ax_phase1.contourf(X1_mesh, X2_mesh, V_vals_12, levels=40, cmap='viridis', alpha=1)
        cbar = fig.colorbar(contour, ax=ax_phase1)
        cbar.set_label('$V(x)$', fontsize=25)

    # Draw projection of regions onto x1-x2 plane
    ax_phase1.add_patch(Rectangle(
        (init_range[0, 0], init_range[1, 0]),
        init_range[0, 1] - init_range[0, 0],
        init_range[1, 1] - init_range[1, 0],
        fill=False, edgecolor=colors['init'], lw=1.5, alpha=0.7, label='Init'
    ))
    ax_phase1.add_patch(Rectangle(
        (goal_range[0, 0], goal_range[1, 0]),
        goal_range[0, 1] - goal_range[0, 0],
        goal_range[1, 1] - goal_range[1, 0],
        fill=False, edgecolor=colors['goal'], lw=1.5, alpha=0.7, label='Goal'
    ))
    for k in range(unsafeK.shape[0]):
        ax_phase1.add_patch(Rectangle(
            (unsafeK[k, 0, 0], unsafeK[k, 1, 0]),
            unsafeK[k, 0, 1] - unsafeK[k, 0, 0],
            unsafeK[k, 1, 1] - unsafeK[k, 1, 0],
            fill=False, edgecolor=colors['unsafe'], lw=1.2, alpha=0.5, label='Unsafe' if k == 0 else None
        ))

    # Phase plane 2: x2 vs x3
    ax_phase2.set_xlabel('$x_2$', fontsize=25)
    ax_phase2.set_ylabel('$x_3$', fontsize=25)
    # ax_phase2.set_title('$x_2$ vs $x_3$', fontsize=25, weight='medium')
    ax_phase2.grid(True, alpha=0.3)
    ax_phase2.set_xlim(float(full_range[1, 0]), float(full_range[1, 1]))
    ax_phase2.set_ylim(float(full_range[2, 0]), float(full_range[2, 1]))

    # Draw V contour if available
    if V_contour_data_23 is not None:
        X2_mesh, X3_mesh, V_vals_23 = V_contour_data_23
        contour = ax_phase2.contourf(X2_mesh, X3_mesh, V_vals_23, levels=40, cmap='viridis', alpha=1)
        cbar = fig.colorbar(contour, ax=ax_phase2)
        cbar.set_label('$V(x)$', fontsize=25)

    # Draw projection of regions onto x2-x3 plane
    ax_phase2.add_patch(Rectangle(
        (init_range[1, 0], init_range[2, 0]),
        init_range[1, 1] - init_range[1, 0],
        init_range[2, 1] - init_range[2, 0],
        fill=False, edgecolor=colors['init'], lw=1.5, alpha=0.7
    ))
    ax_phase2.add_patch(Rectangle(
        (goal_range[1, 0], goal_range[2, 0]),
        goal_range[1, 1] - goal_range[1, 0],
        goal_range[2, 1] - goal_range[2, 0],
        fill=False, edgecolor=colors['goal'], lw=1.5, alpha=0.7
    ))
    for k in range(unsafeK.shape[0]):
        ax_phase2.add_patch(Rectangle(
            (unsafeK[k, 1, 0], unsafeK[k, 2, 0]),
            unsafeK[k, 1, 1] - unsafeK[k, 1, 0],
            unsafeK[k, 2, 1] - unsafeK[k, 2, 0],
            fill=False, edgecolor=colors['unsafe'], lw=1.2, alpha=0.5
        ))

    # Phase plane trajectory lines and points
    phase1_lines_closed = []
    phase1_pts_closed = []
    phase2_lines_closed = []
    phase2_pts_closed = []
    phase1_lines_open = []
    phase1_pts_open = []
    phase2_lines_open = []
    phase2_pts_open = []

    for i in range(n_trajectories):
        # Closed-loop phase trajectories
        line1, = ax_phase1.plot([], [], lw=1.5, color=colors_closed[i], alpha=0.7, label='Controlled' if i == 0 else None)
        pt1, = ax_phase1.plot([], [], marker='o', markersize=4, color=colors_closed[i])
        phase1_lines_closed.append(line1)
        phase1_pts_closed.append(pt1)

        line2, = ax_phase2.plot([], [], lw=1.5, color=colors_closed[i], alpha=0.7)
        pt2, = ax_phase2.plot([], [], marker='o', markersize=4, color=colors_closed[i])
        phase2_lines_closed.append(line2)
        phase2_pts_closed.append(pt2)

        # Open-loop phase trajectories
        if f_open_module is not None:
            line1_open, = ax_phase1.plot([], [], lw=1.5, color=colors_open[i], alpha=0.7, linestyle='--', label='Uncontrolled' if i == 0 else None)
            pt1_open, = ax_phase1.plot([], [], marker='s', markersize=4, color=colors_open[i])
            phase1_lines_open.append(line1_open)
            phase1_pts_open.append(pt1_open)

            line2_open, = ax_phase2.plot([], [], lw=1.5, color=colors_open[i], alpha=0.7, linestyle='--')
            pt2_open, = ax_phase2.plot([], [], marker='s', markersize=4, color=colors_open[i])
            phase2_lines_open.append(line2_open)
            phase2_pts_open.append(pt2_open)

    # Add legend to phase plane 1
    ax_phase1.legend(loc='best', framealpha=0.9, fontsize=12)

    # ========== Animation Functions ==========
    def init_anim():
        artists = []
        # Closed-loop 3D
        for line, pt in zip(lines3d_closed, pts3d_closed):
            line.set_data([], [])
            line.set_3d_properties([])
            pt.set_data([], [])
            pt.set_3d_properties([])
            artists.extend([line, pt])

        # Open-loop 3D
        if f_open_module is not None:
            for line, pt in zip(lines3d_open, pts3d_open):
                line.set_data([], [])
                line.set_3d_properties([])
                pt.set_data([], [])
                pt.set_3d_properties([])
                artists.extend([line, pt])

        # Phase plane lines
        for line, pt in zip(phase1_lines_closed, phase1_pts_closed):
            line.set_data([], [])
            pt.set_data([], [])
            artists.extend([line, pt])
        for line, pt in zip(phase2_lines_closed, phase2_pts_closed):
            line.set_data([], [])
            pt.set_data([], [])
            artists.extend([line, pt])

        if f_open_module is not None:
            for line, pt in zip(phase1_lines_open, phase1_pts_open):
                line.set_data([], [])
                pt.set_data([], [])
                artists.extend([line, pt])
            for line, pt in zip(phase2_lines_open, phase2_pts_open):
                line.set_data([], [])
                pt.set_data([], [])
                artists.extend([line, pt])
        return artists

    def update(i: int):
        i = int(i)
        artists = []

        # Update all closed-loop trajectories (top left 3D plot + phase planes)
        for traj_idx in range(n_trajectories):
            X_traj = all_trajectories_closed[traj_idx]
            U_traj = all_controls_closed[traj_idx]
            t_traj = all_times_closed[traj_idx]

            # Find index for this trajectory at current time
            current_time = t[i]
            if current_time <= t_traj[-1]:
                # Find closest index
                traj_i = int(current_time / dt)
                traj_i = min(traj_i, len(X_traj) - 1)

                # 3D trajectory
                lines3d_closed[traj_idx].set_data(X_traj[:traj_i+1, 0], X_traj[:traj_i+1, 1])
                lines3d_closed[traj_idx].set_3d_properties(X_traj[:traj_i+1, 2])
                pts3d_closed[traj_idx].set_data([X_traj[traj_i, 0]], [X_traj[traj_i, 1]])
                pts3d_closed[traj_idx].set_3d_properties([X_traj[traj_i, 2]])

                # Phase plane 1: x1 vs x2
                phase1_lines_closed[traj_idx].set_data(X_traj[:traj_i+1, 0], X_traj[:traj_i+1, 1])
                phase1_pts_closed[traj_idx].set_data([X_traj[traj_i, 0]], [X_traj[traj_i, 1]])

                # Phase plane 2: x2 vs x3
                phase2_lines_closed[traj_idx].set_data(X_traj[:traj_i+1, 1], X_traj[:traj_i+1, 2])
                phase2_pts_closed[traj_idx].set_data([X_traj[traj_i, 1]], [X_traj[traj_i, 2]])
            else:
                # Trajectory has ended, show final state
                lines3d_closed[traj_idx].set_data(X_traj[:, 0], X_traj[:, 1])
                lines3d_closed[traj_idx].set_3d_properties(X_traj[:, 2])
                pts3d_closed[traj_idx].set_data([X_traj[-1, 0]], [X_traj[-1, 1]])
                pts3d_closed[traj_idx].set_3d_properties([X_traj[-1, 2]])

                phase1_lines_closed[traj_idx].set_data(X_traj[:, 0], X_traj[:, 1])
                phase1_pts_closed[traj_idx].set_data([X_traj[-1, 0]], [X_traj[-1, 1]])

                phase2_lines_closed[traj_idx].set_data(X_traj[:, 1], X_traj[:, 2])
                phase2_pts_closed[traj_idx].set_data([X_traj[-1, 1]], [X_traj[-1, 2]])

            artists.extend([lines3d_closed[traj_idx], pts3d_closed[traj_idx],
                           phase1_lines_closed[traj_idx], phase1_pts_closed[traj_idx],
                           phase2_lines_closed[traj_idx], phase2_pts_closed[traj_idx]])

        # Update all open-loop trajectories (top 3D plot)
        if f_open_module is not None:
            for traj_idx in range(n_trajectories):
                X_traj_open = all_trajectories_open[traj_idx]
                U_traj_open = all_controls_open[traj_idx]
                t_traj_open = all_times_open[traj_idx]

                # Find index for this trajectory at current time
                current_time = t[i]
                if current_time <= t_traj_open[-1]:
                    # Find closest index
                    traj_i = int(current_time / dt)
                    traj_i = min(traj_i, len(X_traj_open) - 1)

                    # 3D trajectory
                    lines3d_open[traj_idx].set_data(X_traj_open[:traj_i+1, 0], X_traj_open[:traj_i+1, 1])
                    lines3d_open[traj_idx].set_3d_properties(X_traj_open[:traj_i+1, 2])
                    pts3d_open[traj_idx].set_data([X_traj_open[traj_i, 0]], [X_traj_open[traj_i, 1]])
                    pts3d_open[traj_idx].set_3d_properties([X_traj_open[traj_i, 2]])

                    # Phase plane 1: x1 vs x2
                    phase1_lines_open[traj_idx].set_data(X_traj_open[:traj_i+1, 0], X_traj_open[:traj_i+1, 1])
                    phase1_pts_open[traj_idx].set_data([X_traj_open[traj_i, 0]], [X_traj_open[traj_i, 1]])

                    # Phase plane 2: x2 vs x3
                    phase2_lines_open[traj_idx].set_data(X_traj_open[:traj_i+1, 1], X_traj_open[:traj_i+1, 2])
                    phase2_pts_open[traj_idx].set_data([X_traj_open[traj_i, 1]], [X_traj_open[traj_i, 2]])
                else:
                    # Trajectory has ended, show final state
                    lines3d_open[traj_idx].set_data(X_traj_open[:, 0], X_traj_open[:, 1])
                    lines3d_open[traj_idx].set_3d_properties(X_traj_open[:, 2])
                    pts3d_open[traj_idx].set_data([X_traj_open[-1, 0]], [X_traj_open[-1, 1]])
                    pts3d_open[traj_idx].set_3d_properties([X_traj_open[-1, 2]])

                    phase1_lines_open[traj_idx].set_data(X_traj_open[:, 0], X_traj_open[:, 1])
                    phase1_pts_open[traj_idx].set_data([X_traj_open[-1, 0]], [X_traj_open[-1, 1]])

                    phase2_lines_open[traj_idx].set_data(X_traj_open[:, 1], X_traj_open[:, 2])
                    phase2_pts_open[traj_idx].set_data([X_traj_open[-1, 1]], [X_traj_open[-1, 2]])

                artists.extend([lines3d_open[traj_idx], pts3d_open[traj_idx],
                               phase1_lines_open[traj_idx], phase1_pts_open[traj_idx],
                               phase2_lines_open[traj_idx], phase2_pts_open[traj_idx]])

        return artists

    # Create animation - use consistent frame count for consistent speed
    target_frames = 150  # Fixed number of frames for consistent animation speed
    frame_skip = max(1, N // target_frames)
    frames = range(0, N, frame_skip)
    ani = FuncAnimation(fig, update, frames=frames, init_func=init_anim,
                       interval=20, blit=False)  # 20ms = 50fps

    if save_path is not None:
        ani.save(save_path, dpi=150, writer='ffmpeg')
        print(f"Animation saved to {save_path}")

    if save_final_frame is not None:
        # Draw final frame as static figure
        update(N - 1)
        fig.savefig(save_final_frame, dpi=300, bbox_inches='tight')
        print(f"Final frame saved to {save_final_frame}")

    if show:
        plt.show()

    return {
        "trajectories_closed": all_trajectories_closed,
        "controls_closed": all_controls_closed,
        "times_closed": all_times_closed,
        "trajectories_open": all_trajectories_open if f_open_module is not None else None,
        "controls_open": all_controls_open if f_open_module is not None else None,
        "times_open": all_times_open if f_open_module is not None else None,
        "n_trajectories": n_trajectories
    }
