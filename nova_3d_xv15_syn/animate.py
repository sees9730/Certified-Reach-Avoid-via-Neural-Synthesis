"""Trajectory animation for the learned XV-15 switching policy.

Run as: python -m nova_3d_xv15_syn anim
Standalone: python -m nova_3d_xv15_syn.animate

Simulates the closed-loop system under the learned policy/certificate pair
saved in results/candidate.json and animates: the 2D (x, z) flight-path
trajectory, the 3D (v, gamma, beta) phase-space trajectory against the
init/goal/unsafe regions, the state time series (v, gamma, beta) shaded
with their projected initial/goal intervals, and the control time series
(thrust, angle of attack, tilt rate).

This module only reads a saved candidate; it does not modify training,
verification, or plotting code.
"""
import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'nova_3d_xv15_syn_matplotlib'))
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

try:
    from .interval import I
    from .model import Model, drift
except ImportError:
    from interval import I
    from model import Model, drift

HERE = Path(__file__).resolve().parent


def simulate(model, p, *, n_trajectories=5, dt=0.02, T=12.0, seed=0):
    """Simulate closed-loop trajectories under the learned switching policy.

    Each trajectory stops as soon as it enters the goal region or an unsafe
    box (or the time horizon T is reached).

    Returns a list of dicts, one per trajectory, each with keys
    't', 'X' (v, gamma[deg], beta[deg]), 'U' (thrust[N], alpha[deg], delta[deg/s]),
    'P' (x, z position from integrating v, gamma), 'outcome' ('goal', 'unsafe', or 'timeout').
    """
    rng = np.random.default_rng(seed)
    domain = np.array(p['domain'], dtype=np.float64)
    init_range = np.array(p['initial'], dtype=np.float64)
    goal_range = np.array(p['goal'], dtype=np.float64)
    unsafe_boxes = np.array(p['unsafe'], dtype=np.float64)
    diffusion = np.array(p['diffusion'], dtype=np.float64)
    radians = np.pi / 180

    def in_goal(x_state):
        return bool(np.all(x_state >= goal_range[:, 0]) and np.all(x_state <= goal_range[:, 1]))

    def in_unsafe(x_state):
        return bool(np.any(np.all(x_state >= unsafe_boxes[:, :, 0], axis=1)
                            & np.all(x_state <= unsafe_boxes[:, :, 1], axis=1)))

    N_max = int(T / dt) + 1
    trajectories = []

    model.eval()
    with torch.no_grad():
        for _ in range(n_trajectories):
            x0 = rng.uniform(init_range[:, 0], init_range[:, 1])
            X = np.zeros((N_max, 3), dtype=np.float64)
            U = np.zeros((N_max, 3), dtype=np.float64)
            X[0] = x0

            N = N_max
            outcome = 'timeout'
            for k in range(N_max - 1):
                xk = I(torch.tensor(X[k], dtype=torch.float64))
                u_list = model.policy(xk)
                xdot_list = drift(xk, u_list, p)
                u = np.array([float(ui.lo) for ui in u_list])
                xdot = np.array([float(xd.lo) for xd in xdot_list])

                dW = np.sqrt(dt) * rng.standard_normal(3)
                xnext = X[k] + dt * xdot + diffusion * dW
                xnext = np.clip(xnext, domain[:, 0], domain[:, 1])

                X[k + 1] = xnext
                U[k] = u

                if in_goal(xnext):
                    outcome = 'goal'
                elif in_unsafe(xnext):
                    outcome = 'unsafe'
                if outcome != 'timeout':
                    N = k + 2
                    X, U = X[:N], U[:N]
                    break
            U[-1] = U[-2]

            P = np.zeros((N, 2), dtype=np.float64)
            for k in range(N - 1):
                v_k, gamma_k = X[k, 0], X[k, 1] * radians
                P[k + 1] = P[k] + dt * np.array([v_k * np.cos(gamma_k), v_k * np.sin(gamma_k)])

            t = np.linspace(0.0, (N - 1) * dt, N, dtype=np.float64)
            U_phys = U.copy()
            U_phys[:, 0] *= p['mass'] * p['gravity']
            trajectories.append(dict(t=t, X=X, U=U_phys, P=P, outcome=outcome))

    return trajectories


def _draw_box3d(ax3d, box3x2, *, lw=1.2, color='k', alpha=0.6, label=None):
    v0, v1 = float(box3x2[0, 0]), float(box3x2[0, 1])
    g0, g1 = float(box3x2[1, 0]), float(box3x2[1, 1])
    b0, b1 = float(box3x2[2, 0]), float(box3x2[2, 1])
    corners = np.array([
        [v0, g0, b0], [v1, g0, b0], [v1, g1, b0], [v0, g1, b0],
        [v0, g0, b1], [v1, g0, b1], [v1, g1, b1], [v0, g1, b1],
    ])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    for i, (e0, e1) in enumerate(edges):
        ax3d.plot([corners[e0, 0], corners[e1, 0]], [corners[e0, 1], corners[e1, 1]],
                  [corners[e0, 2], corners[e1, 2]], lw=lw, color=color, alpha=alpha,
                  label=label if i == 0 else None)


def animate_nova(model, p, *, n_trajectories=5, dt=0.02, T=12.0, seed=0,
                  save_path=None, save_final_frame=None, show=True):
    trajectories = simulate(model, p, n_trajectories=n_trajectories, dt=dt, T=T, seed=seed)
    domain = np.array(p['domain'], dtype=np.float64)
    init_range = np.array(p['initial'], dtype=np.float64)
    goal_range = np.array(p['goal'], dtype=np.float64)
    unsafeK = np.array(p['unsafe'], dtype=np.float64)

    max_len = max(len(traj['t']) for traj in trajectories)
    t = np.linspace(0.0, (max_len - 1) * dt, max_len, dtype=np.float64)
    N = max_len

    plt.rcParams.update({
        'font.size': 13, 'axes.labelsize': 12, 'axes.titlesize': 14,
        'axes.linewidth': 0.8, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
        'legend.fontsize': 10, 'legend.framealpha': 0.9,
        'grid.alpha': 0.3, 'grid.linewidth': 0.5,
    })
    colors = {'init': 'seagreen', 'goal': 'darkgoldenrod', 'unsafe': 'firebrick', 'full': '#95a5a6'}
    traj_colors = plt.cm.plasma(np.linspace(0.15, 0.85, n_trajectories))

    fig = plt.figure(figsize=(17, 10), dpi=110)
    fig.suptitle('XV-15 learned switching policy — closed-loop trajectories', fontsize=15, y=0.98, weight='medium')
    gs = fig.add_gridspec(6, 3, width_ratios=[1.3, 1.0, 1.0], height_ratios=[1, 1, 1, 1, 1, 1],
                          wspace=0.4, hspace=0.6, top=0.92, bottom=0.07, left=0.06, right=0.94)

    ax_pos = fig.add_subplot(gs[:3, 0])
    ax_3d = fig.add_subplot(gs[3:, 0], projection='3d')
    ax_v = fig.add_subplot(gs[0:2, 1])
    ax_gamma = fig.add_subplot(gs[2:4, 1], sharex=ax_v)
    ax_beta = fig.add_subplot(gs[4:6, 1], sharex=ax_v)
    ax_thrust = fig.add_subplot(gs[0:2, 2])
    ax_alpha = fig.add_subplot(gs[2:4, 2], sharex=ax_thrust)
    ax_delta = fig.add_subplot(gs[4:6, 2], sharex=ax_thrust)

    ax_pos.set_xlabel('$x$ [m]')
    ax_pos.set_ylabel('$z$ [m]')
    ax_pos.grid(True, alpha=0.3)
    ax_pos.set_title('Flight-path trajectory (x-z)', weight='medium')
    all_x = np.concatenate([traj['P'][:, 0] for traj in trajectories])
    all_z = np.concatenate([traj['P'][:, 1] for traj in trajectories])
    pad = 0.15
    x_range = max(all_x.max() - all_x.min(), 50.0)
    z_range = max(all_z.max() - all_z.min(), 50.0)
    x_center, z_center = (all_x.max() + all_x.min()) / 2, (all_z.max() + all_z.min()) / 2
    ax_pos.set_xlim(x_center - (1 + pad) * x_range / 2, x_center + (1 + pad) * x_range / 2)
    ax_pos.set_ylim(z_center - (1 + pad) * z_range / 2, z_center + (1 + pad) * z_range / 2)

    arrow_len = 0.06 * max(x_range, z_range)

    pos_lines, pos_vel_arrows, pos_body_arrows = [], [], []
    for i in range(n_trajectories):
        line, = ax_pos.plot([], [], lw=1.5, color=traj_colors[i], alpha=0.8)
        vel_arrow = ax_pos.quiver([0.], [0.], [1.], [0.], color='0.45',
                                  angles='xy', scale_units='xy', scale=1, pivot='tail',
                                  width=0.004, headwidth=3, headlength=4, alpha=0.7, zorder=4)
        body_arrow = ax_pos.quiver([0.], [0.], [1.], [0.], color=traj_colors[i],
                                   angles='xy', scale_units='xy', scale=1, pivot='tail',
                                   width=0.006, headwidth=4, headlength=5, zorder=5)
        pos_lines.append(line); pos_vel_arrows.append(vel_arrow); pos_body_arrows.append(body_arrow)

    ax_pos.quiverkey(pos_vel_arrows[0], 0.02, 0.04, arrow_len, 'Velocity ($\\gamma$)',
                     labelpos='E', coordinates='axes', fontproperties={'size': 9}, color='0.45')
    ax_pos.quiverkey(pos_body_arrows[0], 0.02, 0.12, arrow_len, 'Nose ($\\gamma+\\alpha$) — offset shows AoA',
                     labelpos='E', coordinates='axes', fontproperties={'size': 9})

    _draw_box3d(ax_3d, domain, lw=0.8, color=colors['full'], alpha=0.3)
    _draw_box3d(ax_3d, init_range, lw=1.5, color=colors['init'], alpha=0.8, label='Init')
    _draw_box3d(ax_3d, goal_range, lw=1.5, color=colors['goal'], alpha=0.8, label='Goal')
    for k in range(unsafeK.shape[0]):
        _draw_box3d(ax_3d, unsafeK[k], lw=1.0, color=colors['unsafe'], alpha=0.4,
                   label='Unsafe' if k == 0 else None)

    lines3d, pts3d = [], []
    for i in range(n_trajectories):
        line, = ax_3d.plot([], [], [], lw=1.5, color=traj_colors[i], alpha=0.8)
        pt, = ax_3d.plot([], [], [], marker='o', markersize=4, color=traj_colors[i])
        lines3d.append(line); pts3d.append(pt)

    ax_3d.set_xlabel('$v$ [m/s]', labelpad=6)
    ax_3d.set_ylabel('$\\gamma$ [deg]', labelpad=6)
    ax_3d.set_zlabel('$\\beta$ [deg]', labelpad=6)
    ax_3d.set_xlim(float(domain[0, 0]), float(domain[0, 1]))
    ax_3d.set_ylim(float(domain[1, 0]), float(domain[1, 1]))
    ax_3d.set_zlim(float(domain[2, 0]), float(domain[2, 1]))
    ax_3d.view_init(elev=20, azim=-60)
    ax_3d.legend(loc='lower left', fontsize=9)
    ax_3d.grid(True, alpha=0.2)
    ax_3d.set_title('Phase-space trajectory (v, $\\gamma$, $\\beta$)', weight='medium', y=1.0)

    right_axes = [ax_v, ax_gamma, ax_beta, ax_thrust, ax_alpha, ax_delta]
    right_series = [
        ('X', 0, 'Airspeed $v$ [m/s]', 0),
        ('X', 1, 'Flight-path angle $\\gamma$ [deg]', 1),
        ('X', 2, 'Rotor tilt $\\beta$ [deg]', 2),
        ('U', 0, 'Thrust [N]', None),
        ('U', 1, 'Angle of attack $\\alpha$ [deg]', None),
        ('U', 2, 'Tilt rate $\\delta$ [deg/s]', None),
    ]
    lines_right = [[] for _ in right_axes]
    for ax, (key, dim, label, state_dim) in zip(right_axes, right_series):
        ax.set_ylabel(label, fontsize=10)
        ax.grid(True, alpha=0.3)
        all_v = np.concatenate([traj[key][:, dim] for traj in trajectories])
        v_min, v_max = all_v.min(), all_v.max()
        if state_dim is not None:
            # Include the projected initial/goal interval so bands are never clipped out.
            v_min = min(v_min, init_range[state_dim, 0], goal_range[state_dim, 0])
            v_max = max(v_max, init_range[state_dim, 1], goal_range[state_dim, 1])
        if np.isclose(v_min, v_max):
            pad = max(abs(v_min), 1.0) * 0.1
            ax.set_ylim(v_min - pad, v_max + pad)
        else:
            r = v_max - v_min
            ax.set_ylim(v_min - 0.1 * r, v_max + 0.1 * r)
        if state_dim is not None:
            ax.axhspan(init_range[state_dim, 0], init_range[state_dim, 1],
                      color=colors['init'], alpha=0.15, zorder=0,
                      label='Initial' if state_dim == 0 else None)
            ax.axhspan(goal_range[state_dim, 0], goal_range[state_dim, 1],
                      color=colors['goal'], alpha=0.15, zorder=0,
                      label='Goal' if state_dim == 0 else None)
        for i in range(n_trajectories):
            line, = ax.plot([], [], lw=1.3, color=traj_colors[i], alpha=0.8)
            lines_right[right_axes.index(ax)].append(line)

    ax_v.legend(loc='upper right', fontsize=8, framealpha=0.8)

    max_T = max(traj['t'][-1] for traj in trajectories)
    for ax in right_axes:
        ax.set_xlim(0, max_T)
    ax_v.tick_params(labelbottom=False)
    ax_gamma.tick_params(labelbottom=False)
    ax_beta.set_xlabel('Time [s]')
    ax_thrust.tick_params(labelbottom=False)
    ax_alpha.tick_params(labelbottom=False)
    ax_delta.set_xlabel('Time [s]')

    def init_anim():
        artists = []
        for line, vel_arrow, body_arrow in zip(pos_lines, pos_vel_arrows, pos_body_arrows):
            line.set_data([], [])
            artists += [line, vel_arrow, body_arrow]
        for line, pt in zip(lines3d, pts3d):
            line.set_data([], []); line.set_3d_properties([])
            pt.set_data([], []); pt.set_3d_properties([])
            artists += [line, pt]
        for group in lines_right:
            for line in group:
                line.set_data([], [])
                artists.append(line)
        return artists

    def update(i):
        i = int(i)
        artists = []
        for idx, traj in enumerate(trajectories):
            X_traj, U_traj, P_traj, t_traj = traj['X'], traj['U'], traj['P'], traj['t']
            current_time = min(t[i], t_traj[-1])
            traj_i = min(int(current_time / dt), len(X_traj) - 1)

            pos_lines[idx].set_data(P_traj[:traj_i + 1, 0], P_traj[:traj_i + 1, 1])
            pos_xy = [[P_traj[traj_i, 0], P_traj[traj_i, 1]]]
            gamma_rad = np.radians(X_traj[traj_i, 1])
            pitch_rad = np.radians(X_traj[traj_i, 1] + U_traj[traj_i, 1])
            pos_vel_arrows[idx].set_offsets(pos_xy)
            pos_vel_arrows[idx].set_UVC([arrow_len * np.cos(gamma_rad)], [arrow_len * np.sin(gamma_rad)])
            pos_body_arrows[idx].set_offsets(pos_xy)
            pos_body_arrows[idx].set_UVC([arrow_len * np.cos(pitch_rad)], [arrow_len * np.sin(pitch_rad)])

            lines3d[idx].set_data(X_traj[:traj_i + 1, 0], X_traj[:traj_i + 1, 1])
            lines3d[idx].set_3d_properties(X_traj[:traj_i + 1, 2])
            pts3d[idx].set_data([X_traj[traj_i, 0]], [X_traj[traj_i, 1]])
            pts3d[idx].set_3d_properties([X_traj[traj_i, 2]])

            for group, (key, dim, _, _sd) in zip(lines_right, right_series):
                source = X_traj if key == 'X' else U_traj
                group[idx].set_data(t_traj[:traj_i + 1], source[:traj_i + 1, dim])

            artists += [pos_lines[idx], pos_vel_arrows[idx], pos_body_arrows[idx], lines3d[idx], pts3d[idx]]
            for group in lines_right:
                artists.append(group[idx])
        return artists

    target_frames = 150
    frame_skip = max(1, N // target_frames)
    frames = range(0, N, frame_skip)
    ani = FuncAnimation(fig, update, frames=frames, init_func=init_anim, interval=20, blit=False)

    if save_path is not None:
        save_path = str(save_path)
        if save_path.endswith('.gif'):
            ani.save(save_path, dpi=130, writer='pillow')
        else:
            ani.save(save_path, dpi=130, writer='ffmpeg')
        print(f'Animation saved to {save_path}', flush=True)

    if save_final_frame is not None:
        update(N - 1)
        fig.savefig(save_final_frame, dpi=300, bbox_inches='tight')
        print(f'Final frame saved to {save_final_frame}', flush=True)

    if show:
        plt.show()

    return trajectories, ani


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'results',
                        help='Directory holding candidate.json (default: nova_3d_xv15_syn/results)')
    parser.add_argument('--n-trajectories', type=int, default=5)
    parser.add_argument('--dt', type=float, default=0.02)
    parser.add_argument('--seconds', type=float, default=12.0, help='Simulated horizon T')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save', type=Path, default=None, help='Save animation to this path (.mp4 or .gif)')
    parser.add_argument('--save-final-frame', type=Path, default=None, help='Save the final frame as an image')
    parser.add_argument('--no-show', action='store_true', help='Do not open an interactive window')
    args = parser.parse_args(argv)

    import json
    artifact = json.loads((args.output / 'candidate.json').read_text())
    p = artifact['problem']
    model = Model(p, initialize=False)
    model.restore(artifact['model'])

    animate_nova(model, p, n_trajectories=args.n_trajectories, dt=args.dt, T=args.seconds, seed=args.seed,
                save_path=args.save, save_final_frame=args.save_final_frame, show=not args.no_show)


if __name__ == '__main__':
    main()
