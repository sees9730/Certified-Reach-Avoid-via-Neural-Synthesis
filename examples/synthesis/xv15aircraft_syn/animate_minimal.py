"""
Minimal, publication-ready animation for XV-15 aircraft.
Fewer plots, more expressive visualization.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animate_xv15_minimal(
    *,
    f_cl_module,
    f_open_module=None,
    f_pretrain_module=None,
    g_fn=None,
    init_range: np.ndarray,
    goal_range: np.ndarray,
    full_range: np.ndarray,
    unsafe_boxes: np.ndarray,
    device: str = "cpu",
    dt: float = 0.02,
    T: float = 12.0,
    seed: int = 0,
    save_path: str | None = None,
    save_final_frame: str | None = None,
    show: bool = True,
    controller_label: str | None = None,
    n_trajectories: int = 5,
):
    """
    Minimal, publication-ready animation with only essential plots.
    Focuses on: (1) 3D state trajectory, (2) Combined state/control plot
    Runs multiple stochastic trajectories with different initial conditions.

    Args:
        f_cl_module: Closed-loop dynamics module
        f_open_module: Optional open-loop (non-trained) dynamics module
        f_pretrain_module: Optional pretrained controller dynamics module
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

    # Simulate multiple trajectories with different initial conditions
    N_max = int(T / dt) + 1
    all_trajectories_closed = []
    all_controls_closed = []
    all_positions_closed = []
    all_times_closed = []

    all_trajectories_open = []
    all_controls_open = []
    all_positions_open = []
    all_times_open = []

    all_trajectories_pretrain = []
    all_controls_pretrain = []
    all_positions_pretrain = []
    all_times_pretrain = []

    f_cl_module.eval()
    if f_open_module is not None:
        f_open_module.eval()
    if f_pretrain_module is not None:
        f_pretrain_module.eval()

    # Store initial conditions for reuse in open-loop simulation
    initial_conditions = []

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
                uk = f_cl_module.controller(xk_t).detach().cpu().numpy().reshape(3)
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

        # Compute position for closed-loop trajectory
        P = np.zeros((N, 2), dtype=np.float32)
        P[0] = np.array([0.0, 0.0], dtype=np.float32)
        for k in range(N - 1):
            v_k = float(X[k, 0])
            gamma_k = float(X[k, 1])
            P[k + 1] = P[k] + dt * np.array([v_k * np.cos(gamma_k), v_k * np.sin(gamma_k)], dtype=np.float32)

        t_traj = np.linspace(0.0, (N-1)*dt, N, dtype=np.float32)

        all_trajectories_closed.append(X)
        all_controls_closed.append(U)
        all_positions_closed.append(P)
        all_times_closed.append(t_traj)

    # Find the maximum time across all controlled trajectories
    max_closed_time = max(t_traj[-1] for t_traj in all_times_closed)
    max_closed_steps = int(max_closed_time / dt) + 1
    print(f"Maximum controlled trajectory time: {max_closed_time:.2f}s ({max_closed_steps} steps)")

    # Simulate open-loop trajectories if module provided
    # Run them for longer than the longest controlled trajectory to show divergence
    if f_open_module is not None:
        # Extend open-loop simulation by 50% beyond longest closed-loop time
        open_loop_extension_factor = 2.5
        max_open_time = max_closed_time * open_loop_extension_factor
        max_open_steps = int(max_open_time / dt) + 1

        print(f"Simulating {n_trajectories} open-loop trajectories for {max_open_time:.2f}s...")
        for traj_idx in range(n_trajectories):
            x0 = initial_conditions[traj_idx]

            X_open = np.zeros((max_open_steps, 3), dtype=np.float32)
            U_open = np.zeros((max_open_steps, 3), dtype=np.float32)
            X_open[0] = x0

            with torch.no_grad():
                for k in range(max_open_steps - 1):
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

            # Compute position for open-loop trajectory
            P_open = np.zeros((max_open_steps, 2), dtype=np.float32)
            P_open[0] = np.array([0.0, 0.0], dtype=np.float32)
            for k in range(max_open_steps - 1):
                v_k = float(X_open[k, 0])
                gamma_k = float(X_open[k, 1])
                P_open[k + 1] = P_open[k] + dt * np.array([v_k * np.cos(gamma_k), v_k * np.sin(gamma_k)], dtype=np.float32)

            t_traj_open = np.linspace(0.0, (max_open_steps-1)*dt, max_open_steps, dtype=np.float32)

            all_trajectories_open.append(X_open)
            all_controls_open.append(U_open)
            all_positions_open.append(P_open)
            all_times_open.append(t_traj_open)

    # Simulate pretrained trajectories if module provided
    # Run them for the same duration as longest closed-loop trajectory
    if f_pretrain_module is not None:
        # Use the maximum closed-loop time for pretrained simulations
        max_pretrain_steps = max_closed_steps

        print(f"Simulating {n_trajectories} pretrained trajectories for {max_closed_time:.2f}s...")
        for traj_idx in range(n_trajectories):
            x0 = initial_conditions[traj_idx]

            X_pretrain = np.zeros((max_pretrain_steps, 3), dtype=np.float32)
            U_pretrain = np.zeros((max_pretrain_steps, 3), dtype=np.float32)
            X_pretrain[0] = x0

            with torch.no_grad():
                for k in range(max_pretrain_steps - 1):
                    xk_t = torch.tensor(X_pretrain[k:k + 1], dtype=torch.float32, device=device)
                    uk = f_pretrain_module.controller(xk_t).detach().cpu().numpy().reshape(3)
                    xdot = f_pretrain_module(xk_t).detach().cpu().numpy().reshape(3)

                    if g_fn is None:
                        xnext = X_pretrain[k] + dt * xdot
                    else:
                        gk = g_fn(xk_t).detach().cpu().numpy().reshape(3)
                        dW = (np.sqrt(dt) * rng.standard_normal(3)).astype(np.float32)
                        xnext = X_pretrain[k] + dt * xdot + gk * dW

                    X_pretrain[k + 1] = xnext
                    U_pretrain[k] = uk

                U_pretrain[-1] = U_pretrain[-2]

            # Compute position for pretrained trajectory
            P_pretrain = np.zeros((max_pretrain_steps, 2), dtype=np.float32)
            P_pretrain[0] = np.array([0.0, 0.0], dtype=np.float32)
            for k in range(max_pretrain_steps - 1):
                v_k = float(X_pretrain[k, 0])
                gamma_k = float(X_pretrain[k, 1])
                P_pretrain[k + 1] = P_pretrain[k] + dt * np.array([v_k * np.cos(gamma_k), v_k * np.sin(gamma_k)], dtype=np.float32)

            t_traj_pretrain = np.linspace(0.0, (max_pretrain_steps-1)*dt, max_pretrain_steps, dtype=np.float32)

            all_trajectories_pretrain.append(X_pretrain)
            all_controls_pretrain.append(U_pretrain)
            all_positions_pretrain.append(P_pretrain)
            all_times_pretrain.append(t_traj_pretrain)

    # Use the longest trajectory (open-loop if available, otherwise closed-loop) for animation timing
    # This determines how long the position and 3D plots will animate
    if f_open_module is not None and len(all_trajectories_open) > 0:
        max_len = max(len(traj) for traj in all_trajectories_open)
    else:
        max_len = max(len(traj) for traj in all_trajectories_closed)
    t = np.linspace(0.0, (max_len-1)*dt, max_len, dtype=np.float32)
    N = max_len

    # Clean styling
    plt.rcParams.update({
        'font.size': 15,
        'axes.labelsize': 12,
        'font.family': 'Times New Roman',
        'axes.titlesize': 16,
        'axes.linewidth': 0.8,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 14,
        'legend.framealpha': 0.9,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
    })

    # Colors matching src/visualization.py
    colors = {
        'init': 'seagreen',      # green (matches visualization.py)
        'goal': 'darkgoldenrod',         # blue (matches visualization.py)
        'unsafe': 'firebrick',    # red (matches visualization.py)
        'traj': '#34495e',      # dark gray
        'full': '#95a5a6',      # light gray
    }

    # Colors: use distinct colormaps that are visible against dark background
    cmap_open = plt.cm.binary  # Binary shades for uncontrolled
    cmap_closed = plt.cm.inferno  # Inferno shades for controlled

    # Generate colors from colormaps (use brighter end of spectrum)
    # colors_open = [cmap_open(0.3 + 0.1 * i) for i in range(n_trajectories)]
    # colors_closed = [cmap_closed(0.3 + 0.1 * i) for i in range(n_trajectories)]
    # colors_closed = ["darkgoldenrod" for i in range(n_trajectories)]
    # colors_open = ["grey" for i in range(n_trajectories)]
    colors_open = ["black"] * n_trajectories
    colors_closed = ["deeppink"] * n_trajectories
    colors_pretrain = ["steelblue"] * n_trajectories  # Different color for pretrained

    # Layout: Aircraft position (top left), 3D state (bottom left), 3 state+control plots (right)
    fig = plt.figure(figsize=(14, 8), dpi=120)

    if controller_label:
        fig.suptitle(controller_label, fontsize=16, y=0.97, weight='medium')

    gs = fig.add_gridspec(3, 2, width_ratios=[1.2, 1.0],
                          height_ratios=[1, 1, 1],
                          wspace=0.35, hspace=0.30,
                          top=0.94, bottom=0.08, left=0.08, right=0.88)

    ax_pos = fig.add_subplot(gs[0, 0])
    ax_3d = fig.add_subplot(gs[1:, 0], projection="3d")
    ax_v = fig.add_subplot(gs[0, 1])
    ax_gamma = fig.add_subplot(gs[1, 1], sharex=ax_v)
    ax_beta = fig.add_subplot(gs[2, 1], sharex=ax_v)

    # ========== TOP LEFT: Aircraft Position (x-z) ==========
    ax_pos.set_xlabel('$x$ [m]', fontsize=20)
    ax_pos.set_ylabel('$z$ [m]',fontsize=20)
    ax_pos.grid(True, alpha=0.3)
    ax_pos.set_title('Aircraft Position', fontsize=20, weight='medium')

    # Set limits based on all trajectories with padding
    pad = 0.15
    all_x = np.concatenate([P[:, 0] for P in all_positions_closed])
    all_z = np.concatenate([P[:, 1] for P in all_positions_closed])
    if f_open_module is not None:
        all_x = np.concatenate([all_x] + [P[:, 0] for P in all_positions_open])
        all_z = np.concatenate([all_z] + [P[:, 1] for P in all_positions_open])
    if f_pretrain_module is not None:
        all_x = np.concatenate([all_x] + [P[:, 0] for P in all_positions_pretrain])
        all_z = np.concatenate([all_z] + [P[:, 1] for P in all_positions_pretrain])
    x_range = all_x.max() - all_x.min()
    z_range = all_z.max() - all_z.min()

    # Ensure minimum range to prevent excessive shrinking
    min_range = 50.0  # minimum 50m range
    x_range = max(x_range, min_range)
    z_range = max(z_range, min_range)

    x_center = (all_x.max() + all_x.min()) / 2
    z_center = (all_z.max() + all_z.min()) / 2

    ax_pos.set_xlim(x_center - (1 + pad) * x_range / 2, x_center + (1 + pad) * x_range / 2)
    ax_pos.set_ylim(z_center - (1 + pad) * z_range / 2, z_center + (1 + pad) * z_range / 2)

    # Position trajectory lines and points for each trajectory
    pos_lines_closed = []
    pos_pts_closed = []
    pos_lines_open = []
    pos_pts_open = []
    pos_lines_pretrain = []
    pos_pts_pretrain = []

    for i in range(n_trajectories):
        # Closed-loop trajectories
        # Add label only for first trajectory to appear once in legend
        label_closed = 'Controlled' if i == 0 else None
        line, = ax_pos.plot([], [], lw=1.5, color=colors_closed[i], alpha=0.7, label=label_closed)
        pt, = ax_pos.plot([], [], marker='o', markersize=5, color=colors_closed[i])
        pos_lines_closed.append(line)
        pos_pts_closed.append(pt)

        # Open-loop trajectories
        if f_open_module is not None:
            label_open = 'Uncontrolled' if i == 0 else None
            line_open, = ax_pos.plot([], [], lw=1.5, color=colors_open[i], alpha=0.7, linestyle='--', label=label_open)
            pt_open, = ax_pos.plot([], [], marker='s', markersize=5, color=colors_open[i])
            pos_lines_open.append(line_open)
            pos_pts_open.append(pt_open)

        # Pretrained trajectories
        if f_pretrain_module is not None:
            label_pretrain = 'Pretrained' if i == 0 else None
            line_pretrain, = ax_pos.plot([], [], lw=1.0, color=colors_pretrain[i], alpha=0.3, label=label_pretrain)
            pt_pretrain, = ax_pos.plot([], [], marker='d', markersize=4, color=colors_pretrain[i], alpha=0.3)
            pos_lines_pretrain.append(line_pretrain)
            pos_pts_pretrain.append(pt_pretrain)

    # Add legend to position plot
    # ax_pos.legend(loc='best', framealpha=0.9, fontsize=16)

    # ========== BOTTOM LEFT: 3D State Trajectory ==========
    def _draw_box3d(ax3d, box3x2, *, lw=1.2, color="k", alpha=0.6, label=None):
        v0, v1 = float(box3x2[0, 0]), float(box3x2[0, 1])
        g0, g1 = float(box3x2[1, 0]), float(box3x2[1, 1])
        b0, b1 = float(box3x2[2, 0]), float(box3x2[2, 1])

        corners = np.array([
            [v0, g0, b0], [v1, g0, b0], [v1, g1, b0], [v0, g1, b0],
            [v0, g0, b1], [v1, g0, b1], [v1, g1, b1], [v0, g1, b1],
        ])

        edges = [
            (0,1),(1,2),(2,3),(3,0),  # bottom
            (4,5),(5,6),(6,7),(7,4),  # top
            (0,4),(1,5),(2,6),(3,7),  # verticals
        ]
        for i, (e0, e1) in enumerate(edges):
            ax3d.plot([corners[e0, 0], corners[e1, 0]],
                     [corners[e0, 1], corners[e1, 1]],
                     [corners[e0, 2], corners[e1, 2]],
                     lw=lw, color=color, alpha=alpha,
                     label=label if i == 0 else None)

    # Draw regions with colors matching src/visualization.py
    _draw_box3d(ax_3d, full_range, lw=0.8, color=colors['full'], alpha=0.3)
    _draw_box3d(ax_3d, init_range, lw=1.5, color=colors['init'], alpha=0.7, label='Init')
    _draw_box3d(ax_3d, goal_range, lw=1.5, color=colors['goal'], alpha=0.7, label='Goal')

    for k in range(unsafeK.shape[0]):
        _draw_box3d(ax_3d, unsafeK[k], lw=1.2, color=colors['unsafe'], alpha=0.5,
                   label='Unsafe' if k == 0 else None)

    # Trajectory lines and points for each trajectory in 3D
    lines3d_closed = []
    pts3d_closed = []
    lines3d_open = []
    pts3d_open = []
    lines3d_pretrain = []
    pts3d_pretrain = []

    for i in range(n_trajectories):
        # Closed-loop trajectories
        line, = ax_3d.plot([], [], [], lw=1.5, color=colors_closed[i], alpha=0.7)
        pt, = ax_3d.plot([], [], [], marker='o', markersize=4, color=colors_closed[i])
        lines3d_closed.append(line)
        pts3d_closed.append(pt)

        # Open-loop trajectories
        if f_open_module is not None:
            line_open, = ax_3d.plot([], [], [], lw=1.5, color=colors_open[i], alpha=0.7, linestyle='--')
            pt_open, = ax_3d.plot([], [], [], marker='s', markersize=4, color=colors_open[i])
            lines3d_open.append(line_open)
            pts3d_open.append(pt_open)

        # Pretrained trajectories
        if f_pretrain_module is not None:
            line_pretrain, = ax_3d.plot([], [], [], lw=1.0, color=colors_pretrain[i], alpha=0.3)
            pt_pretrain, = ax_3d.plot([], [], [], marker='d', markersize=3, color=colors_pretrain[i], alpha=0.3)
            lines3d_pretrain.append(line_pretrain)
            pts3d_pretrain.append(pt_pretrain)

    # Styling
    ax_3d.set_xlabel('$v$ [m/s]', labelpad=8, fontsize=20)
    ax_3d.set_ylabel('$\\gamma$ [rad]', labelpad=8, fontsize=20)
    ax_3d.set_zlabel('$\\beta$ [rad]', labelpad=8, fontsize=20)
    ax_3d.set_xlim(float(full_range[0, 0]), float(full_range[0, 1]))
    ax_3d.set_ylim(float(full_range[1, 0]), float(full_range[1, 1]))
    ax_3d.set_zlim(float(full_range[2, 0]), float(full_range[2, 1]))
    ax_3d.view_init(elev=20, azim=-60)
    ax_3d.legend(loc='upper left', framealpha=0.9, fontsize=16)
    ax_3d.grid(True, alpha=0.2)

    # ========== RIGHT: Three subplots for control inputs only ==========
    axes = [ax_v, ax_gamma, ax_beta]
    ctrl_names = ['$T$ [N]', '$\\alpha$ [rad]', '$\\delta$ [rad/s]']

    # Control lines for each trajectory
    ctrl_lines_closed = []   # Will be list of lists: ctrl_lines_closed[traj_idx][ctrl_dim]
    ctrl_lines_open = []     # Will be list of lists: ctrl_lines_open[traj_idx][ctrl_dim]
    ctrl_lines_pretrain = [] # Will be list of lists: ctrl_lines_pretrain[traj_idx][ctrl_dim]

    # Setup each subplot
    for i, (ax, ctrl_name) in enumerate(zip(axes, ctrl_names)):
        # Styling
        ax.set_ylabel(ctrl_name, fontsize=20)
        ax.grid(True, alpha=0.3)

        # Set y-limits based on all trajectory data
        pad = 0.1
        all_u = np.concatenate([U[:, i] for U in all_controls_closed])
        if f_open_module is not None:
            all_u = np.concatenate([all_u] + [U[:, i] for U in all_controls_open])
        if f_pretrain_module is not None:
            all_u = np.concatenate([all_u] + [U[:, i] for U in all_controls_pretrain])
        u_min, u_max = all_u.min(), all_u.max()
        if not np.isclose(u_min, u_max):
            ctrl_range = u_max - u_min
            ax.set_ylim(u_min - pad * ctrl_range, u_max + pad * ctrl_range)

    # Create control lines for each trajectory
    for traj_idx in range(n_trajectories):
        # Closed-loop control lines
        traj_ctrl_lines = []
        for i, ax in enumerate(axes):
            # Add label only on first trajectory and only on first (torque) plot
            label_closed = 'Bound Trained SAT' if (traj_idx == 0 and i == 0) else None
            line, = ax.plot([], [], lw=1.5, color=colors_closed[traj_idx], alpha=0.7, label=label_closed)
            traj_ctrl_lines.append(line)
        ctrl_lines_closed.append(traj_ctrl_lines)

        # Open-loop control lines
        if f_open_module is not None:
            traj_ctrl_lines_open = []
            for i, ax in enumerate(axes):
                # Add label only on first trajectory and only on first (torque) plot
                label_open = 'Uncontrolled' if (traj_idx == 0 and i == 0) else None
                line_open, = ax.plot([], [], lw=1.5, color=colors_open[traj_idx], alpha=0.7, linestyle='--', label=label_open)
                traj_ctrl_lines_open.append(line_open)
            ctrl_lines_open.append(traj_ctrl_lines_open)

        # Pretrained control lines (smaller alpha and thinner line)
        if f_pretrain_module is not None:
            traj_ctrl_lines_pretrain = []
            for i, ax in enumerate(axes):
                # Add label only on first trajectory and only on first (torque) plot
                label_pretrain = 'Warm-Start UNSAT' if (traj_idx == 0 and i == 0) else None
                line_pretrain, = ax.plot([], [], lw=1.0, color=colors_pretrain[traj_idx], alpha=0.2, label=label_pretrain)
                traj_ctrl_lines_pretrain.append(line_pretrain)
            ctrl_lines_pretrain.append(traj_ctrl_lines_pretrain)

    # Set x-limits for control plots to longest controlled trajectory time
    actual_T_controlled = float(max_closed_time)
    for ax in axes:
        ax.set_xlim(0, actual_T_controlled)

    # Add legend to the torque plot (first plot)
    ax_v.legend(loc='best', framealpha=0.9, fontsize=16)

    # Hide x-tick labels for top two plots only
    ax_v.tick_params(labelbottom=False)
    ax_gamma.tick_params(labelbottom=False)
    ax_beta.tick_params(labelbottom=True)

    # Only bottom plot gets x-label
    ax_beta.set_xlabel('Time [s]', fontsize=20)

    fig.tight_layout()

    # ========== Animation Functions ==========
    def init_anim():
        artists = []
        # Closed-loop
        for line, pt in zip(pos_lines_closed, pos_pts_closed):
            line.set_data([], [])
            pt.set_data([], [])
            artists.extend([line, pt])
        for line, pt in zip(lines3d_closed, pts3d_closed):
            line.set_data([], [])
            line.set_3d_properties([])
            pt.set_data([], [])
            pt.set_3d_properties([])
            artists.extend([line, pt])
        for traj_ctrl_lines in ctrl_lines_closed:
            for line in traj_ctrl_lines:
                line.set_data([], [])
                artists.append(line)

        # Open-loop
        if f_open_module is not None:
            for line, pt in zip(pos_lines_open, pos_pts_open):
                line.set_data([], [])
                pt.set_data([], [])
                artists.extend([line, pt])
            for line, pt in zip(lines3d_open, pts3d_open):
                line.set_data([], [])
                line.set_3d_properties([])
                pt.set_data([], [])
                pt.set_3d_properties([])
                artists.extend([line, pt])
            for traj_ctrl_lines in ctrl_lines_open:
                for line in traj_ctrl_lines:
                    line.set_data([], [])
                    artists.append(line)

        # Pretrained
        if f_pretrain_module is not None:
            for line, pt in zip(pos_lines_pretrain, pos_pts_pretrain):
                line.set_data([], [])
                pt.set_data([], [])
                artists.extend([line, pt])
            for line, pt in zip(lines3d_pretrain, pts3d_pretrain):
                line.set_data([], [])
                line.set_3d_properties([])
                pt.set_data([], [])
                pt.set_3d_properties([])
                artists.extend([line, pt])
            for traj_ctrl_lines in ctrl_lines_pretrain:
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
            P_traj = all_positions_closed[traj_idx]
            t_traj = all_times_closed[traj_idx]

            # Find index for this trajectory at current time
            current_time = t[i]
            if current_time <= t_traj[-1]:
                # Find closest index
                traj_i = int(current_time / dt)
                traj_i = min(traj_i, len(X_traj) - 1)

                # Position trajectory
                pos_lines_closed[traj_idx].set_data(P_traj[:traj_i+1, 0], P_traj[:traj_i+1, 1])
                pos_pts_closed[traj_idx].set_data([P_traj[traj_i, 0]], [P_traj[traj_i, 1]])

                # 3D trajectory
                lines3d_closed[traj_idx].set_data(X_traj[:traj_i+1, 0], X_traj[:traj_i+1, 1])
                lines3d_closed[traj_idx].set_3d_properties(X_traj[:traj_i+1, 2])
                pts3d_closed[traj_idx].set_data([X_traj[traj_i, 0]], [X_traj[traj_i, 1]])
                pts3d_closed[traj_idx].set_3d_properties([X_traj[traj_i, 2]])

                # Control time series
                for ctrl_dim in range(3):
                    ctrl_lines_closed[traj_idx][ctrl_dim].set_data(t_traj[:traj_i+1], U_traj[:traj_i+1, ctrl_dim])
            else:
                # Trajectory has ended, show final state
                pos_lines_closed[traj_idx].set_data(P_traj[:, 0], P_traj[:, 1])
                pos_pts_closed[traj_idx].set_data([P_traj[-1, 0]], [P_traj[-1, 1]])

                lines3d_closed[traj_idx].set_data(X_traj[:, 0], X_traj[:, 1])
                lines3d_closed[traj_idx].set_3d_properties(X_traj[:, 2])
                pts3d_closed[traj_idx].set_data([X_traj[-1, 0]], [X_traj[-1, 1]])
                pts3d_closed[traj_idx].set_3d_properties([X_traj[-1, 2]])

                for ctrl_dim in range(3):
                    ctrl_lines_closed[traj_idx][ctrl_dim].set_data(t_traj, U_traj[:, ctrl_dim])

            artists.extend([pos_lines_closed[traj_idx], pos_pts_closed[traj_idx],
                           lines3d_closed[traj_idx], pts3d_closed[traj_idx]])
            artists.extend(ctrl_lines_closed[traj_idx])

        # Update all open-loop trajectories
        if f_open_module is not None:
            for traj_idx in range(n_trajectories):
                X_traj_open = all_trajectories_open[traj_idx]
                U_traj_open = all_controls_open[traj_idx]
                P_traj_open = all_positions_open[traj_idx]
                t_traj_open = all_times_open[traj_idx]

                # Find index for this trajectory at current time
                current_time = t[i]
                if current_time <= t_traj_open[-1]:
                    # Find closest index
                    traj_i = int(current_time / dt)
                    traj_i = min(traj_i, len(X_traj_open) - 1)

                    # Position trajectory
                    pos_lines_open[traj_idx].set_data(P_traj_open[:traj_i+1, 0], P_traj_open[:traj_i+1, 1])
                    pos_pts_open[traj_idx].set_data([P_traj_open[traj_i, 0]], [P_traj_open[traj_i, 1]])

                    # 3D trajectory
                    lines3d_open[traj_idx].set_data(X_traj_open[:traj_i+1, 0], X_traj_open[:traj_i+1, 1])
                    lines3d_open[traj_idx].set_3d_properties(X_traj_open[:traj_i+1, 2])
                    pts3d_open[traj_idx].set_data([X_traj_open[traj_i, 0]], [X_traj_open[traj_i, 1]])
                    pts3d_open[traj_idx].set_3d_properties([X_traj_open[traj_i, 2]])

                    # Control time series
                    for ctrl_dim in range(3):
                        ctrl_lines_open[traj_idx][ctrl_dim].set_data(t_traj_open[:traj_i+1], U_traj_open[:traj_i+1, ctrl_dim])
                else:
                    # Trajectory has ended, show final state
                    pos_lines_open[traj_idx].set_data(P_traj_open[:, 0], P_traj_open[:, 1])
                    pos_pts_open[traj_idx].set_data([P_traj_open[-1, 0]], [P_traj_open[-1, 1]])

                    lines3d_open[traj_idx].set_data(X_traj_open[:, 0], X_traj_open[:, 1])
                    lines3d_open[traj_idx].set_3d_properties(X_traj_open[:, 2])
                    pts3d_open[traj_idx].set_data([X_traj_open[-1, 0]], [X_traj_open[-1, 1]])
                    pts3d_open[traj_idx].set_3d_properties([X_traj_open[-1, 2]])

                    for ctrl_dim in range(3):
                        ctrl_lines_open[traj_idx][ctrl_dim].set_data(t_traj_open, U_traj_open[:, ctrl_dim])

                artists.extend([pos_lines_open[traj_idx], pos_pts_open[traj_idx],
                               lines3d_open[traj_idx], pts3d_open[traj_idx]])
                artists.extend(ctrl_lines_open[traj_idx])

        # Update all pretrained trajectories (position, 3D, and control plots)
        if f_pretrain_module is not None:
            for traj_idx in range(n_trajectories):
                X_traj_pretrain = all_trajectories_pretrain[traj_idx]
                U_traj_pretrain = all_controls_pretrain[traj_idx]
                P_traj_pretrain = all_positions_pretrain[traj_idx]
                t_traj_pretrain = all_times_pretrain[traj_idx]

                # Find index for this trajectory at current time
                current_time = t[i]
                if current_time <= t_traj_pretrain[-1]:
                    # Find closest index
                    traj_i = int(current_time / dt)
                    traj_i = min(traj_i, len(X_traj_pretrain) - 1)

                    # Position trajectory
                    pos_lines_pretrain[traj_idx].set_data(P_traj_pretrain[:traj_i+1, 0], P_traj_pretrain[:traj_i+1, 1])
                    pos_pts_pretrain[traj_idx].set_data([P_traj_pretrain[traj_i, 0]], [P_traj_pretrain[traj_i, 1]])

                    # 3D trajectory
                    lines3d_pretrain[traj_idx].set_data(X_traj_pretrain[:traj_i+1, 0], X_traj_pretrain[:traj_i+1, 1])
                    lines3d_pretrain[traj_idx].set_3d_properties(X_traj_pretrain[:traj_i+1, 2])
                    pts3d_pretrain[traj_idx].set_data([X_traj_pretrain[traj_i, 0]], [X_traj_pretrain[traj_i, 1]])
                    pts3d_pretrain[traj_idx].set_3d_properties([X_traj_pretrain[traj_i, 2]])

                    # Control time series
                    for ctrl_dim in range(3):
                        ctrl_lines_pretrain[traj_idx][ctrl_dim].set_data(t_traj_pretrain[:traj_i+1], U_traj_pretrain[:traj_i+1, ctrl_dim])
                else:
                    # Trajectory has ended, show final state
                    pos_lines_pretrain[traj_idx].set_data(P_traj_pretrain[:, 0], P_traj_pretrain[:, 1])
                    pos_pts_pretrain[traj_idx].set_data([P_traj_pretrain[-1, 0]], [P_traj_pretrain[-1, 1]])

                    lines3d_pretrain[traj_idx].set_data(X_traj_pretrain[:, 0], X_traj_pretrain[:, 1])
                    lines3d_pretrain[traj_idx].set_3d_properties(X_traj_pretrain[:, 2])
                    pts3d_pretrain[traj_idx].set_data([X_traj_pretrain[-1, 0]], [X_traj_pretrain[-1, 1]])
                    pts3d_pretrain[traj_idx].set_3d_properties([X_traj_pretrain[-1, 2]])

                    for ctrl_dim in range(3):
                        ctrl_lines_pretrain[traj_idx][ctrl_dim].set_data(t_traj_pretrain, U_traj_pretrain[:, ctrl_dim])

                artists.extend([pos_lines_pretrain[traj_idx], pos_pts_pretrain[traj_idx],
                               lines3d_pretrain[traj_idx], pts3d_pretrain[traj_idx]])
                artists.extend(ctrl_lines_pretrain[traj_idx])

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
        "positions_closed": all_positions_closed,
        "times_closed": all_times_closed,
        "trajectories_open": all_trajectories_open if f_open_module is not None else None,
        "controls_open": all_controls_open if f_open_module is not None else None,
        "positions_open": all_positions_open if f_open_module is not None else None,
        "times_open": all_times_open if f_open_module is not None else None,
        "trajectories_pretrain": all_trajectories_pretrain if f_pretrain_module is not None else None,
        "controls_pretrain": all_controls_pretrain if f_pretrain_module is not None else None,
        "positions_pretrain": all_positions_pretrain if f_pretrain_module is not None else None,
        "times_pretrain": all_times_pretrain if f_pretrain_module is not None else None,
        "n_trajectories": n_trajectories
    }
