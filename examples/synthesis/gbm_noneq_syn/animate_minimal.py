"""
Minimal, publication-ready animation for 2D GBM System.
Layout: 2D phase plot (left) + 2 control plots (right).
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle


def animate_gbm_minimal(
    *,
    f_cl_module,
    f_open_module=None,
    g_fn=None,
    V_net=None,
    init_range: dict,
    goal_range: dict,
    full_range: dict,
    unsafe_range: dict,
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
    Focuses on: (1) 2D phase plot, (2) Control plots

    Args:
        f_cl_module: Closed-loop dynamics module
        f_open_module: Optional open-loop (non-trained) dynamics module
        g_fn: Optional diffusion function (returns 2x2 matrix)
        V_net: Optional value network for contour overlay
        init_range: Initial region bounds dict
        goal_range: Goal region bounds dict
        full_range: Full state space bounds dict
        unsafe_range: Unsafe region bounds dict
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

    def in_box(x, bounds_dict):
        """Check if state x is in box defined by bounds_dict."""
        return (bounds_dict["x1_min"] <= x[0] <= bounds_dict["x1_max"] and
                bounds_dict["x2_min"] <= x[1] <= bounds_dict["x2_max"])

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
            rng.uniform(init_range["x1_min"], init_range["x1_max"]),
            rng.uniform(init_range["x2_min"], init_range["x2_max"]),
        ], dtype=np.float32)
        initial_conditions.append(x0)

        # Simulate closed-loop (controlled) trajectory
        X = np.zeros((N_max, 2), dtype=np.float32)
        U = np.zeros((N_max, 2), dtype=np.float32)
        X[0] = x0

        N = N_max
        goal_reached_step = None
        extra_steps_after_goal = int(0.5 / dt)  # 2 seconds worth of steps

        with torch.no_grad():
            for k in range(N_max - 1):
                xk_t = torch.tensor(X[k:k + 1], dtype=torch.float32, device=device)

                # Get control from controller if available
                if hasattr(f_cl_module, 'controller'):
                    uk = f_cl_module.controller(xk_t).detach().cpu().numpy().reshape(2)
                else:
                    uk = np.zeros(2, dtype=np.float32)

                # Get closed-loop drift (f + u)
                xdot = f_cl_module(xk_t).detach().cpu().numpy().reshape(2)

                if g_fn is None:
                    xnext = X[k] + dt * xdot
                else:
                    gk = g_fn(xk_t).detach().cpu().numpy().reshape(2, 2)
                    dW = (np.sqrt(dt) * rng.standard_normal(2)).astype(np.float32)
                    xnext = X[k] + dt * xdot + gk @ dW

                X[k + 1] = xnext
                U[k] = uk

                # Track when goal is first reached
                if goal_reached_step is None and in_box(xnext, goal_range):
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

    # For open-loop, simulate longer to reach equilibrium
    # Add extra time beyond controlled trajectories
    extra_time_for_open_loop = 8.0  # Extra seconds for open-loop to settle
    max_open_steps = int((max_closed_time + extra_time_for_open_loop) / dt) + 1

    print(f"Maximum controlled trajectory time: {max_closed_time:.2f}s")
    print(f"Open-loop simulation time: {(max_open_steps-1)*dt:.2f}s")

    # Simulate open-loop trajectories if module provided (for longer time)
    if f_open_module is not None:
        print(f"Simulating {n_trajectories} open-loop trajectories...")
        for traj_idx in range(n_trajectories):
            x0 = initial_conditions[traj_idx]

            X_open = np.zeros((max_open_steps, 2), dtype=np.float32)
            U_open = np.zeros((max_open_steps, 2), dtype=np.float32)
            X_open[0] = x0

            with torch.no_grad():
                for k in range(max_open_steps - 1):
                    xk_t = torch.tensor(X_open[k:k + 1], dtype=torch.float32, device=device)
                    xdot_open = f_open_module(xk_t).detach().cpu().numpy().reshape(2)

                    if g_fn is None:
                        xnext_open = X_open[k] + dt * xdot_open
                    else:
                        gk = g_fn(xk_t).detach().cpu().numpy().reshape(2, 2)
                        dW = (np.sqrt(dt) * rng.standard_normal(2)).astype(np.float32)
                        xnext_open = X_open[k] + dt * xdot_open + gk @ dW

                    X_open[k + 1] = xnext_open
                    # Open-loop has zero control
                    U_open[k] = np.zeros(2, dtype=np.float32)

                U_open[-1] = U_open[-2]

            t_traj_open = np.linspace(0.0, (max_open_steps-1)*dt, max_open_steps, dtype=np.float32)

            all_trajectories_open.append(X_open)
            all_controls_open.append(U_open)
            all_times_open.append(t_traj_open)

    # Use the longest trajectory for animation timing (open-loop is longer)
    if f_open_module is not None:
        max_len = max(len(traj) for traj in all_trajectories_open)
    else:
        max_len = max(len(traj) for traj in all_trajectories_closed)
    t = np.linspace(0.0, (max_len-1)*dt, max_len, dtype=np.float32)
    N = max_len

    # Store where controlled trajectories end (for control plot limits)
    max_closed_len = max(len(traj) for traj in all_trajectories_closed)
    max_closed_time_actual = (max_closed_len - 1) * dt

    # Clean styling
    plt.rcParams.update({
        'font.size': 15,
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix',  # Use STIX fonts for math (similar to Times)
        'axes.labelsize': 18,
        'axes.titlesize': 25,
        'axes.linewidth': 0.8,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 11,
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

    # Colors: use distinct colormaps
    cmap_open = plt.cm.Greys
    cmap_closed = plt.cm.inferno

    # Generate colors from colormaps
    # colors_open = [cmap_open(0.5 + 0.1 * i) for i in range(n_trajectories)]
    # colors_closed = [cmap_closed(0.4 + 0.15 * i) for i in range(n_trajectories)]
    colors_open = ["black"] * n_trajectories
    colors_closed = ["deeppink"] * n_trajectories

    # Layout: 2D phase plot (left), 2 control plots (right)
    fig = plt.figure(figsize=(12, 6), dpi=120)

    # if controller_label:
    #     fig.suptitle(controller_label, fontsize=17, y=0.96, weight='medium')

    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1.0],
                          height_ratios=[1, 1],
                          wspace=0.30, hspace=0.25,
                          top=0.92, bottom=0.10, left=0.08, right=0.92)

    ax_phase = fig.add_subplot(gs[:, 0])
    ax_u1 = fig.add_subplot(gs[0, 1])
    ax_u2 = fig.add_subplot(gs[1, 1], sharex=ax_u1)

    # ========== LEFT: 2D Phase Plot ==========
    ax_phase.set_xlabel('$x_1$', fontsize=15)
    ax_phase.set_ylabel('$x_2$', fontsize=15)
    ax_phase.set_title('Phase Plane', fontsize=15, weight='medium')
    ax_phase.grid(False)  # No background grid
    ax_phase.set_xlim(full_range["x1_min"], full_range["x1_max"])
    ax_phase.set_ylim(full_range["x2_min"], full_range["x2_max"])

    # Plot V(x) contours if V_net is provided
    if V_net is not None:
        # Create grid for contour plot
        n_grid = 100
        x1_grid = np.linspace(full_range["x1_min"], full_range["x1_max"], n_grid)
        x2_grid = np.linspace(full_range["x2_min"], full_range["x2_max"], n_grid)
        X1_mesh, X2_mesh = np.meshgrid(x1_grid, x2_grid)

        # Evaluate V on grid
        V_net.eval()
        with torch.no_grad():
            grid_points = torch.tensor(
                np.stack([X1_mesh.ravel(), X2_mesh.ravel()], axis=1),
                dtype=torch.float32,
                device=device
            )
            V_values = V_net(grid_points).detach().cpu().numpy().reshape(n_grid, n_grid)

        # Plot filled contours with transparency
        contour = ax_phase.contourf(X1_mesh, X2_mesh, V_values, levels=20,
                                     cmap='viridis', alpha=1, zorder=0)
        # Add contour lines for V level sets
        ax_phase.contour(X1_mesh, X2_mesh, V_values, levels=10,
                        colors='gray', alpha=0.4, linewidths=0.5, zorder=0)
        # Add colorbar
        cbar = fig.colorbar(contour, ax=ax_phase, pad=0.02, fraction=0.046)
        cbar.set_label('$V(x)$', fontsize=15)

    # Draw region boxes (unfilled, just borders)
    ax_phase.add_patch(Rectangle(
        (full_range["x1_min"], full_range["x2_min"]),
        full_range["x1_max"] - full_range["x1_min"],
        full_range["x2_max"] - full_range["x2_min"],
        fill=False, edgecolor=colors['full'], lw=1.5, linestyle='-'
    ))
    ax_phase.add_patch(Rectangle(
        (init_range["x1_min"], init_range["x2_min"]),
        init_range["x1_max"] - init_range["x1_min"],
        init_range["x2_max"] - init_range["x2_min"],
        fill=False, edgecolor=colors['init'], lw=1.5, linestyle='-', label='Init'
    ))
    ax_phase.add_patch(Rectangle(
        (goal_range["x1_min"], goal_range["x2_min"]),
        goal_range["x1_max"] - goal_range["x1_min"],
        goal_range["x2_max"] - goal_range["x2_min"],
        fill=False, edgecolor=colors['goal'], lw=1.5, linestyle='-', label='Goal'
    ))
    ax_phase.add_patch(Rectangle(
        (unsafe_range["x1_min"], unsafe_range["x2_min"]),
        unsafe_range["x1_max"] - unsafe_range["x1_min"],
        unsafe_range["x2_max"] - unsafe_range["x2_min"],
        fill=False, edgecolor=colors['unsafe'], lw=1.5, linestyle='-', label='Unsafe'
    ))

    # Trajectory lines and points
    lines_closed = []
    pts_closed = []
    lines_open = []
    pts_open = []

    for i in range(n_trajectories):
        # Closed-loop trajectories
        label_closed = 'Controlled' if i == 0 else None
        line, = ax_phase.plot([], [], lw=1.5, color=colors_closed[i], alpha=0.7, label=label_closed)
        pt, = ax_phase.plot([], [], marker='o', markersize=5, color=colors_closed[i])
        lines_closed.append(line)
        pts_closed.append(pt)

        # Open-loop trajectories
        if f_open_module is not None:
            label_open = 'Uncontrolled' if i == 0 else None
            line_open, = ax_phase.plot([], [], lw=1.5, color=colors_open[i], alpha=0.7,
                                        linestyle='--', label=label_open)
            pt_open, = ax_phase.plot([], [], marker='s', markersize=5, color=colors_open[i])
            lines_open.append(line_open)
            pts_open.append(pt_open)

    ax_phase.legend(loc='best', framealpha=0.9, fontsize=12)

    # ========== RIGHT: Two subplots for control inputs ==========
    axes = [ax_u1, ax_u2]
    ctrl_names = ['$u_1$', '$u_2$']

    # Control lines for each trajectory
    ctrl_lines_closed = []
    ctrl_lines_open = []

    # Setup each subplot
    for i, (ax, ctrl_name) in enumerate(zip(axes, ctrl_names)):
        # Styling
        ax.set_ylabel(ctrl_name, fontweight='medium', fontsize=15)
        ax.grid(True, alpha=0.3)

        # Set y-limits based on all trajectory data
        pad = 0.1
        all_u = np.concatenate([U[:, i] for U in all_controls_closed])
        if f_open_module is not None:
            all_u = np.concatenate([all_u] + [U[:, i] for U in all_controls_open])
        u_min, u_max = all_u.min(), all_u.max()
        if not np.isclose(u_min, u_max):
            ctrl_range = u_max - u_min
            ax.set_ylim(u_min - pad * ctrl_range, u_max + pad * ctrl_range)

    # Create control lines for each trajectory
    for traj_idx in range(n_trajectories):
        # Closed-loop control lines
        traj_ctrl_lines = []
        for i, ax in enumerate(axes):
            # Add label only to first trajectory in first plot (u1) for legend
            label_closed = 'Controlled' if (traj_idx == 0 and i == 0) else None
            line, = ax.plot([], [], lw=1.5, color=colors_closed[traj_idx], alpha=0.7, label=label_closed)
            traj_ctrl_lines.append(line)
        ctrl_lines_closed.append(traj_ctrl_lines)

        # Open-loop control lines
        if f_open_module is not None:
            traj_ctrl_lines_open = []
            for i, ax in enumerate(axes):
                # Add label only to first trajectory in first plot (u1) for legend
                label_open = 'Uncontrolled' if (traj_idx == 0 and i == 0) else None
                line_open, = ax.plot([], [], lw=1.5, color=colors_open[traj_idx], alpha=0.7,
                                     linestyle='--', label=label_open)
                traj_ctrl_lines_open.append(line_open)
            ctrl_lines_open.append(traj_ctrl_lines_open)

    # Add legend to u1 plot only
    ax_u1.legend(loc='best', framealpha=0.9, fontsize=12)

    # Set x-limits to controlled trajectory time (not the longer open-loop time)
    # This avoids showing long flat control signals
    for ax in axes:
        ax.set_xlim(0, max_closed_time_actual)

    # Hide x-tick labels for top plot
    ax_u1.tick_params(labelbottom=False)
    ax_u2.tick_params(labelbottom=True)

    # Only bottom plot gets x-label
    ax_u2.set_xlabel('Time [s]', fontsize=15)

    # ========== Animation Functions ==========
    def init_anim():
        artists = []
        # Closed-loop
        for line, pt in zip(lines_closed, pts_closed):
            line.set_data([], [])
            pt.set_data([], [])
            artists.extend([line, pt])
        for traj_ctrl_lines in ctrl_lines_closed:
            for line in traj_ctrl_lines:
                line.set_data([], [])
                artists.append(line)

        # Open-loop
        if f_open_module is not None:
            for line, pt in zip(lines_open, pts_open):
                line.set_data([], [])
                pt.set_data([], [])
                artists.extend([line, pt])
            for traj_ctrl_lines in ctrl_lines_open:
                for line in traj_ctrl_lines:
                    line.set_data([], [])
                    artists.append(line)
        return artists

    def update(i: int):
        i = int(i)
        artists = []

        # Update all closed-loop trajectories
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

                # Phase trajectory
                lines_closed[traj_idx].set_data(X_traj[:traj_i+1, 0], X_traj[:traj_i+1, 1])
                pts_closed[traj_idx].set_data([X_traj[traj_i, 0]], [X_traj[traj_i, 1]])

                # Control time series
                for ctrl_dim in range(2):
                    ctrl_lines_closed[traj_idx][ctrl_dim].set_data(t_traj[:traj_i+1], U_traj[:traj_i+1, ctrl_dim])
            else:
                # Trajectory has ended, show final state
                lines_closed[traj_idx].set_data(X_traj[:, 0], X_traj[:, 1])
                pts_closed[traj_idx].set_data([X_traj[-1, 0]], [X_traj[-1, 1]])

                for ctrl_dim in range(2):
                    ctrl_lines_closed[traj_idx][ctrl_dim].set_data(t_traj, U_traj[:, ctrl_dim])

            artists.extend([lines_closed[traj_idx], pts_closed[traj_idx]])
            artists.extend(ctrl_lines_closed[traj_idx])

        # Update all open-loop trajectories
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

                    # Phase trajectory
                    lines_open[traj_idx].set_data(X_traj_open[:traj_i+1, 0], X_traj_open[:traj_i+1, 1])
                    pts_open[traj_idx].set_data([X_traj_open[traj_i, 0]], [X_traj_open[traj_i, 1]])

                    # Control time series
                    for ctrl_dim in range(2):
                        ctrl_lines_open[traj_idx][ctrl_dim].set_data(t_traj_open[:traj_i+1], U_traj_open[:traj_i+1, ctrl_dim])
                else:
                    # Trajectory has ended, show final state
                    lines_open[traj_idx].set_data(X_traj_open[:, 0], X_traj_open[:, 1])
                    pts_open[traj_idx].set_data([X_traj_open[-1, 0]], [X_traj_open[-1, 1]])

                    for ctrl_dim in range(2):
                        ctrl_lines_open[traj_idx][ctrl_dim].set_data(t_traj_open, U_traj_open[:, ctrl_dim])

                artists.extend([lines_open[traj_idx], pts_open[traj_idx]])
                artists.extend(ctrl_lines_open[traj_idx])

        return artists

    # Create animation - use consistent frame count for consistent speed
    target_frames = 150
    frame_skip = max(1, N // target_frames)
    frames = range(0, N, frame_skip)
    ani = FuncAnimation(fig, update, frames=frames, init_func=init_anim,
                       interval=20, blit=False)

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
