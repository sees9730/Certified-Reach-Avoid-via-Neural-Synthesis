"""
Two-Room Temperature Control SDE Benchmark
============================================

Continuous-time stochastic differential equation model for temperature
regulation in two adjacent rooms, adapted from the ten-room thermal
building model in:

    P. Jagtap, S. Soudjani, and M. Zamani,
    "Temporal Logic Verification of Stochastic Systems Using Barrier
     Certificates," ATVA 2018, LNCS 11138, pp. 177--193.

and its journal extension:

    P. Jagtap, S. Soudjani, and M. Zamani,
    "Formal Synthesis of Stochastic Systems via Control Barrier
     Certificates," IEEE Trans. Automat. Control, 66(7):3097--3110, 2021.


System Dynamics
---------------
The state x = (x1, x2) represents the temperatures (°C) of two adjacent
rooms that exchange heat with each other and with the outdoor environment.
Each room has an independent heater input.

The continuous-time SDE is

    dx = f(x, u; lambda) dt + g dw(t),

where

    f(x, u; lambda) = A @ x + B @ u + E * lambda,
    g               = sigma * I_2,
    w(t)            = 2-dimensional standard Brownian motion.

Matrices:

    A  = [[ -(alpha + alpha_e),   alpha            ],
          [  alpha,               -(alpha + alpha_e) ]]

    B  = [[ beta,  0    ],
          [  0,    beta  ]]

    E  = [ alpha_e,  alpha_e ]^T

    lambda = T_e    (ambient / external temperature)


Physical Parameters
-------------------
    alpha    : float  —  Inter-room heat transfer coefficient [1/s].
                         Models conduction through the shared interior wall.
                         Reference value: 5e-2.

    alpha_e  : float  —  Room-to-exterior heat loss coefficient [1/s].
                         Models heat loss through exterior walls, windows, etc.
                         Reference value: 5e-3.

    beta     : float  —  Heater effectiveness [°C / (s · input-unit)].
                         Converts control input into temperature rise rate.
                         Reference value: 0.1.

    T_e      : float  —  Ambient (external) temperature [°C].
                         Nominal value: 10.0.

    sigma    : float  —  Diffusion (noise) intensity [°C / sqrt(s)].
                         Captures unmodeled thermal disturbances such as
                         occupancy, solar radiation, door openings, etc.
                         Reference value: 0.5.


Uncertainty Model
-----------------
The drift is affine in the uncertain parameter lambda = T_e:

    f(x, u; lambda) = f_nom(x, u) + E * lambda,

where
    f_nom(x, u)  =  A @ x  +  B @ u  +  E * T_e_nom

and lambda represents the deviation from the nominal ambient temperature,
or equivalently the full ambient temperature treated as uncertain.

Two uncertainty configurations are provided:

1.  **Additive disturbance (unknown ambient temperature)**
        lambda = T_e  in  [T_e_min, T_e_max]  (a box in R^1).
    Since f is affine in lambda over a 1D box, Proposition 1 of the
    main paper yields the robust generator in closed form without
    Lambda-partitioning.

2.  **Set-valued unknown drift (uncertain thermal parameters)**
        lambda = (alpha, alpha_e)  in  [alpha_min, alpha_max] x [alpha_e_min, alpha_e_max].
    The drift is also affine in (alpha, alpha_e) for each fixed (x, u),
    so Proposition 1 again applies.


Reach-Avoid Specification
-------------------------
    X   = [x_min, x_max]^2          Domain of interest.
    X0  = [x0_lo, x0_hi]^2          Initial set   (both rooms cold).
    Xg  = [xg_lo, xg_hi]^2          Goal set      (comfort zone).
    Xu  = X \\ int(Xs)               Unsafe set    (too hot or too cold).
    Xs  = [xs_lo, xs_hi]^2           Safe set      (acceptable range).

Default values:
    X   = [10, 30]^2
    X0  = [17, 19]^2
    Xg  = [20, 22]^2
    Xs  = [15, 25]^2
    Xu  = X \\ int(Xs)                i.e., any room below 15 or above 25.

The reach-avoid task: starting from X0, steer both rooms into the
comfort zone Xg while keeping both temperatures within Xs at all times.


Control Constraints
-------------------
    u_i  in  [-u_max, u_max],    i = 1, 2.

Default: u_max = 1.0  (signed HVAC effort).
The control set U = [-u_max, u_max]^2 is compact.
"""

import argparse

import numpy as np


# ---------------------------------------------------------------------------
#  Physical parameters
# ---------------------------------------------------------------------------
ALPHA = 5e-2        # inter-room heat transfer coefficient  [1/s]
ALPHA_E = 5e-3      # room-to-exterior heat loss coefficient [1/s]
BETA = 0.1          # heater effectiveness                   [°C/(s·u)]
T_E_NOM = 20.0      # nominal ambient temperature            [°C]
SIGMA = 0.05         # diffusion intensity                    [°C/sqrt(s)]

# ---------------------------------------------------------------------------
#  System matrices
# ---------------------------------------------------------------------------
A_DRIFT = np.array([
    [-(ALPHA + ALPHA_E),  ALPHA              ],
    [ ALPHA,             -(ALPHA + ALPHA_E)  ],
])

B_INPUT = np.array([
    [BETA, 0.0 ],
    [0.0,  BETA],
])

E_PARAM = np.array([ALPHA_E, ALPHA_E])   # multiplies lambda = T_e

G_DIFF = SIGMA * np.eye(2)               # diffusion matrix


# ---------------------------------------------------------------------------
#  Uncertainty sets
# ---------------------------------------------------------------------------
# Config 1: uncertain ambient temperature
T_E_MIN = 19.0
T_E_MAX = 21.0

# Config 2: uncertain thermal parameters
ALPHA_RANGE = (4e-2, 6e-2)
ALPHA_E_RANGE = (3e-3, 7e-3)


# ---------------------------------------------------------------------------
#  Reach-avoid sets
# ---------------------------------------------------------------------------
X_DOMAIN = (10.0, 30.0)     # [x_min, x_max] per dimension
X0_INIT = (17.0, 23.0)      # initial set [lo, hi] per dimension
XG_GOAL = (19.0, 21.0)      # goal (comfort) set per dimension
XS_SAFE = (11.0, 29.0)      # safe set per dimension
U_MAX = 3.0                 # max heater input per room


# ---------------------------------------------------------------------------
#  Drift function
# ---------------------------------------------------------------------------
def drift(x, u, T_e=T_E_NOM):
    """
    Evaluate the drift  f(x, u; T_e) = A @ x + B @ u + E * T_e.

    Parameters
    ----------
    x   : ndarray, shape (2,)   —  room temperatures [°C].
    u   : ndarray, shape (2,)   —  heater inputs in [0, u_max].
    T_e : float                 —  ambient temperature [°C].

    Returns
    -------
    f   : ndarray, shape (2,)   —  drift vector.
    """
    return A_DRIFT @ x + B_INPUT @ u + E_PARAM * T_e


def diffusion():
    """
    Return the constant diffusion matrix  g = sigma * I_2.

    Returns
    -------
    g : ndarray, shape (2, 2).
    """
    return G_DIFF.copy()


# ---------------------------------------------------------------------------
#  Robust generator (Proposition 1, closed-form for affine-in-lambda)
# ---------------------------------------------------------------------------
def robust_generator_ambient(grad_V, x, u):
    """
    Compute the worst-case generator contribution from uncertain T_e.

    For the affine structure  f(x,u; T_e) = f_nom(x,u) + E * T_e  with
    T_e in [T_e_min, T_e_max], Proposition 1 gives:

        sup_{T_e}  grad_V^T E * T_e  =  c * T_e_c  +  |c| * T_e_r,

    where c = grad_V^T E, T_e_c = (T_e_min+T_e_max)/2, and
    T_e_r = (T_e_max-T_e_min)/2.

    Parameters
    ----------
    grad_V : ndarray, shape (2,)  —  gradient of V at x.
    x      : ndarray, shape (2,)  —  state (unused, kept for interface).
    u      : ndarray, shape (2,)  —  control (unused, kept for interface).

    Returns
    -------
    sup_val : float  —  sup_{T_e in Lambda}  grad_V^T E * T_e.
    """
    c = grad_V @ E_PARAM
    T_e_c = (T_E_MIN + T_E_MAX) / 2.0
    T_e_r = (T_E_MAX - T_E_MIN) / 2.0
    return c * T_e_c + np.abs(c) * T_e_r


# ---------------------------------------------------------------------------
#  Euler--Maruyama simulation
# ---------------------------------------------------------------------------
def simulate_euler_maruyama(x0, policy, T_horizon, dt=1e-3, T_e=T_E_NOM,
                            rng=None):
    """
    Simulate the closed-loop SDE using the Euler--Maruyama method.

    Parameters
    ----------
    x0        : ndarray, shape (2,)   —  initial state.
    policy    : callable(x) -> u      —  feedback controller.
    T_horizon : float                 —  simulation horizon [s].
    dt        : float                 —  time step [s].
    T_e       : float                 —  ambient temperature [°C].
    rng       : numpy Generator       —  random number generator.

    Returns
    -------
    ts : ndarray, shape (N+1,)       —  time stamps.
    xs : ndarray, shape (N+1, 2)     —  state trajectory.
    us : ndarray, shape (N, 2)       —  applied controls.
    """
    if rng is None:
        rng = np.random.default_rng()

    N_steps = int(np.ceil(T_horizon / dt))
    ts = np.linspace(0.0, T_horizon, N_steps + 1)
    xs = np.zeros((N_steps + 1, 2))
    us = np.zeros((N_steps, 2))
    xs[0] = x0

    sqrt_dt = np.sqrt(dt)
    g = diffusion()

    for k in range(N_steps):
        u = np.clip(policy(xs[k]), -U_MAX, U_MAX)
        us[k] = u
        dw = rng.standard_normal(2) * sqrt_dt
        xs[k + 1] = xs[k] + drift(xs[k], u, T_e) * dt + g @ dw

    return ts, xs, us


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def in_set(x, lo, hi):
    """Check whether x is inside the box [lo, hi]^n (component-wise)."""
    return np.all(x >= lo) and np.all(x <= hi)


def is_safe(x):
    """Check whether state x is in the safe set Xs."""
    return in_set(x, XS_SAFE[0], XS_SAFE[1])


def reached_goal(x):
    """Check whether state x is in the goal set Xg."""
    return in_set(x, XG_GOAL[0], XG_GOAL[1])


def check_reach_avoid(xs):
    """
    Evaluate reach-avoid on a trajectory.

    Parameters
    ----------
    xs : ndarray, shape (N+1, 2)  —  state trajectory.

    Returns
    -------
    success    : bool   —  True if goal reached while staying safe.
    reach_time : float  —  first time the goal is reached (inf if never).
    """
    for k, x in enumerate(xs):
        if not is_safe(x):
            return False, np.inf
        if reached_goal(x):
            return True, k
    return False, np.inf


def bang_bang_policy(x):
    """
    Simple signed bang-bang policy:
    heat if below goal midpoint, cool if above.
    """
    target = (XG_GOAL[0] + XG_GOAL[1]) / 2.0
    return np.array([
        U_MAX if x[0] < target else -U_MAX,
        U_MAX if x[1] < target else -U_MAX,
    ])


def classify_trajectory(xs):
    """
    Classify one trajectory for reach-avoid.
    Returns (status, step), where status is one of {"goal", "unsafe", "timeout"}.
    """
    for k, x in enumerate(xs):
        if not is_safe(x):
            return "unsafe", k
        if reached_goal(x):
            return "goal", k
    return "timeout", len(xs) - 1


def first_goal_hit_step(xs):
    """
    Return the first index k such that xs[k] is in X_goal, else None.
    """
    for k, x in enumerate(xs):
        if reached_goal(x):
            return k
    return None


def run_monte_carlo(policy, n_mc, T_horizon, dt, T_e, rng, store_rollouts=False):
    """
    Run MC rollouts once and return both statistics and (optionally) paths.
    """
    statuses = []
    reach_times = []
    energies = []
    energies_to_hit = []
    paths = []
    controls = []
    energy_hist = []
    hit_steps = []
    ts_ref = None

    for _ in range(n_mc):
        x0 = rng.uniform(X0_INIT[0], X0_INIT[1], size=2)
        ts, xs, us = simulate_euler_maruyama(x0, policy, T_horizon=T_horizon, dt=dt, T_e=T_e, rng=rng)
        status, step = classify_trajectory(xs)
        statuses.append(status)
        hit_steps.append(int(step))

        goal_step = first_goal_hit_step(xs)
        if goal_step is not None:
            bu_hit = us[:goal_step] @ B_INPUT.T
            energy_hit = float(np.sum(bu_hit ** 2) * dt)
            energies_to_hit.append(energy_hit)
            reach_times.append(goal_step * dt)
            energies.append(energy_hit)
        else:
            energies_to_hit.append(np.nan)

        if store_rollouts:
            paths.append(xs)
            controls.append(us)
            bu_full = us @ B_INPUT.T
            power_full = np.sum(bu_full ** 2, axis=1)
            e = np.zeros(xs.shape[0], dtype=float)
            e[1:] = np.cumsum(power_full) * dt
            energy_hist.append(e)
            ts_ref = ts

    out = {
        "statuses": np.array(statuses, dtype=object),
        "hit_steps": np.array(hit_steps, dtype=int),
        "reach_times": np.array(reach_times, dtype=float),
        "energies": np.array(energies, dtype=float),
        "energies_to_hit": np.array(energies_to_hit, dtype=float),
    }
    if store_rollouts:
        out["paths"] = np.stack(paths, axis=0)
        out["controls"] = np.stack(controls, axis=0)
        out["energy_hist"] = np.stack(energy_hist, axis=0)
        out["ts"] = ts_ref
    return out


def animate_mc_results(paths, controls, energy_hist, statuses, hit_steps, ts, skip=3, max_traj=80, save_path=None, show=True):
    """
    Animate a subset of MC trajectories on the phase plane.
    """
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from matplotlib.patches import Rectangle
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for animation. Install it with `pip install matplotlib`."
        ) from exc

    n_total = int(paths.shape[0])
    n_show = min(int(max_traj), n_total)
    if n_show <= 0:
        return None

    if n_total > n_show:
        sel = np.linspace(0, n_total - 1, n_show, dtype=int)
    else:
        sel = np.arange(n_total, dtype=int)

    p = paths[sel]
    u = controls[sel]
    e = energy_hist[sel]
    s = np.asarray(statuses, dtype=object)[sel]
    h = np.asarray(hit_steps, dtype=int)[sel]
    N = p.shape[1]
    N_u = u.shape[1]
    frame_step = max(1, int(skip))

    colors = {"goal": "#2a9d8f", "unsafe": "#e76f51", "timeout": "#888888"}
    t_u = ts[:-1]

    fig = plt.figure(figsize=(12, 6.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.0], height_ratios=[1.0, 1.0])
    ax = fig.add_subplot(gs[:, 0])
    ax_u = fig.add_subplot(gs[0, 1])
    ax_e = fig.add_subplot(gs[1, 1], sharex=ax_u)

    x_min, x_max = X_DOMAIN
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(x_min, x_max)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("MC State Trajectories (Phase Plane)")
    ax.grid(True, alpha=0.25)

    ax.add_patch(Rectangle((x_min, x_min), x_max - x_min, x_max - x_min, fill=False, lw=1.5, color="black"))
    ax.add_patch(Rectangle(
        (XS_SAFE[0], XS_SAFE[0]),
        XS_SAFE[1] - XS_SAFE[0],
        XS_SAFE[1] - XS_SAFE[0],
        facecolor="#d9f2d9",
        edgecolor="#3a7f3a",
        alpha=0.35,
        lw=1.2,
    ))
    ax.add_patch(Rectangle(
        (XG_GOAL[0], XG_GOAL[0]),
        XG_GOAL[1] - XG_GOAL[0],
        XG_GOAL[1] - XG_GOAL[0],
        facecolor="#c9ddff",
        edgecolor="#2f5da8",
        alpha=0.7,
        lw=1.2,
    ))

    lines, points = [], []
    for i in range(n_show):
        c = colors.get(str(s[i]), "#555555")
        ln, = ax.plot([], [], color=c, lw=1.0, alpha=0.65)
        pt, = ax.plot([], [], marker="o", ms=3, color=c, alpha=0.9)
        lines.append(ln)
        points.append(pt)

    ax_u.set_title("Control Channels vs Time")
    ax_u.set_ylabel("u1, u2")
    ax_u.set_xlim(0.0, float(ts[-1]))
    ax_u.set_ylim(-0.05, U_MAX + 0.05)
    ax_u.grid(True, alpha=0.25)
    ax_u.plot([], [], color="black", lw=1.2, linestyle="-", label="u1")
    ax_u.plot([], [], color="black", lw=1.2, linestyle="--", label="u2")
    ax_u.legend(loc="upper right", fontsize=8)

    u1_lines, u1_points = [], []
    u2_lines, u2_points = [], []
    for i in range(n_show):
        c = colors.get(str(s[i]), "#555555")
        ln1, = ax_u.plot([], [], color=c, lw=1.0, alpha=0.65, linestyle="-")
        pt1, = ax_u.plot([], [], marker="o", ms=3, color=c, alpha=0.9)
        ln2, = ax_u.plot([], [], color=c, lw=1.0, alpha=0.65, linestyle="--")
        pt2, = ax_u.plot([], [], marker="s", ms=3, color=c, alpha=0.9)
        u1_lines.append(ln1)
        u1_points.append(pt1)
        u2_lines.append(ln2)
        u2_points.append(pt2)

    ax_e.set_title("Energy vs Time")
    ax_e.set_xlabel("time [s]")
    ax_e.set_ylabel(r"$E(t)=\int_0^t \|Bu\|_2^2 d\tau$")
    ax_e.set_xlim(0.0, float(ts[-1]))
    e_max = float(np.max(e)) if e.size > 0 else 1.0
    ax_e.set_ylim(0.0, max(1.0, 1.05 * e_max))
    ax_e.grid(True, alpha=0.25)

    e_lines, e_points = [], []
    for i in range(n_show):
        c = colors.get(str(s[i]), "#555555")
        ln, = ax_e.plot([], [], color=c, lw=1.0, alpha=0.65)
        pt, = ax_e.plot([], [], marker="o", ms=3, color=c, alpha=0.9)
        e_lines.append(ln)
        e_points.append(pt)

    text_time = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top")
    text_stat = ax.text(0.02, 0.90, "", transform=ax.transAxes, va="top")

    def init_anim():
        for ln, pt, ln_u1, pt_u1, ln_u2, pt_u2, ln_e, pt_e in zip(
            lines, points, u1_lines, u1_points, u2_lines, u2_points, e_lines, e_points
        ):
            ln.set_data([], [])
            pt.set_data([], [])
            ln_u1.set_data([], [])
            pt_u1.set_data([], [])
            ln_u2.set_data([], [])
            pt_u2.set_data([], [])
            ln_e.set_data([], [])
            pt_e.set_data([], [])
        text_time.set_text("")
        text_stat.set_text("")
        return (
            lines + points
            + u1_lines + u1_points + u2_lines + u2_points
            + e_lines + e_points
            + [text_time, text_stat]
        )

    def update(frame_idx):
        for i, (ln, pt, ln_u1, pt_u1, ln_u2, pt_u2, ln_e, pt_e) in enumerate(
            zip(lines, points, u1_lines, u1_points, u2_lines, u2_points, e_lines, e_points)
        ):
            k = min(int(frame_idx), int(h[i]))
            k_u = min(k, N_u)

            ln.set_data(p[i, :k + 1, 0], p[i, :k + 1, 1])
            pt.set_data([p[i, k, 0]], [p[i, k, 1]])

            if k_u > 0:
                ln_u1.set_data(t_u[:k_u], u[i, :k_u, 0])
                pt_u1.set_data([t_u[k_u - 1]], [u[i, k_u - 1, 0]])
                ln_u2.set_data(t_u[:k_u], u[i, :k_u, 1])
                pt_u2.set_data([t_u[k_u - 1]], [u[i, k_u - 1, 1]])
            else:
                ln_u1.set_data([], [])
                pt_u1.set_data([], [])
                ln_u2.set_data([], [])
                pt_u2.set_data([], [])

            ln_e.set_data(ts[:k + 1], e[i, :k + 1])
            pt_e.set_data([ts[k]], [e[i, k]])

        n_goal = int(np.sum([(s[i] == "goal") and (h[i] <= frame_idx) for i in range(n_show)]))
        n_unsafe = int(np.sum([(s[i] == "unsafe") and (h[i] <= frame_idx) for i in range(n_show)]))
        n_alive = n_show - n_goal - n_unsafe

        text_time.set_text(f"t = {float(ts[frame_idx]):.2f} s")
        text_stat.set_text(f"shown={n_show}  goal={n_goal}  unsafe={n_unsafe}  active={n_alive}")
        return (
            lines + points
            + u1_lines + u1_points + u2_lines + u2_points
            + e_lines + e_points
            + [text_time, text_stat]
        )

    ani = FuncAnimation(
        fig,
        update,
        frames=range(0, N, frame_step),
        init_func=init_anim,
        interval=35,
        blit=True,
        repeat=False,
    )
    plt.tight_layout()

    if save_path:
        save_path_str = str(save_path)
        writer = "pillow" if save_path_str.lower().endswith(".gif") else None
        ani.save(save_path_str, dpi=140, writer=writer)
        print(f"Saved animation to: {save_path_str}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return ani


# ---------------------------------------------------------------------------
#  Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-room temperature SDE demo and animation")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    parser.add_argument("--n-mc", type=int, default=100, help="Number of Monte Carlo trajectories.")
    parser.add_argument("--mc-horizon", type=float, default=50.0, help="Monte Carlo horizon [s].")
    parser.add_argument("--mc-dt", type=float, default=0.01, help="Monte Carlo integration step [s].")
    parser.add_argument("--animate-mc", action="store_true", help="Animate subset of MC rollouts (state, control, energy).")
    parser.add_argument("--anim-trajs", type=int, default=80, help="How many MC trajectories to animate.")
    parser.add_argument("--anim-skip", type=int, default=3, help="Frame stride for animation.")
    parser.add_argument("--save-animation", type=str, default=None, help="Optional path to save animation (.gif recommended).")
    parser.add_argument("--no-show", action="store_true", help="Do not open an interactive animation window.")
    args = parser.parse_args()

    print("Two-Room Temperature Control SDE Benchmark")
    print("=" * 50)
    print()
    print("Drift matrix A:")
    print(A_DRIFT)
    print()
    print("Input matrix B:")
    print(B_INPUT)
    print()
    print(f"Ambient temperature range: [{T_E_MIN}, {T_E_MAX}] °C")
    print(f"Diffusion intensity:       {SIGMA} °C/sqrt(s)")
    print()
    print(f"Initial set X0: [{X0_INIT[0]}, {X0_INIT[1]}]^2")
    print(f"Goal set    Xg: [{XG_GOAL[0]}, {XG_GOAL[1]}]^2")
    print(f"Safe set    Xs: [{XS_SAFE[0]}, {XS_SAFE[1]}]^2")
    print()

    rng = np.random.default_rng(int(args.seed))
    n_mc = int(args.n_mc)
    dt = float(args.mc_dt)
    T_horizon = float(args.mc_horizon)
    need_paths = bool(args.animate_mc or args.save_animation)

    mc = run_monte_carlo(
        policy=bang_bang_policy,
        n_mc=n_mc,
        T_horizon=T_horizon,
        dt=dt,
        T_e=T_E_NOM,
        rng=rng,
        store_rollouts=need_paths,
    )

    statuses = mc["statuses"]
    reach_times = mc["reach_times"]
    energies = mc["energies"]
    energies_to_hit = mc["energies_to_hit"]
    n_success = int(np.sum(statuses == "goal"))
    n_unsafe = int(np.sum(statuses == "unsafe"))
    n_timeout = int(np.sum(statuses == "timeout"))

    print(f"Monte Carlo reach-avoid ({n_mc} trials, T_horizon={T_horizon} s):")
    print(f"  Success rate:  {n_success / n_mc:.3f}")
    print(f"  Unsafe rate:   {n_unsafe / n_mc:.3f}")
    print(f"  Timeout rate:  {n_timeout / n_mc:.3f}")
    if np.any(np.isfinite(energies_to_hit)):
        avg_e_hit = float(np.nanmean(energies_to_hit))
        print(f"  Avg energy to t_hit using B@u (goal-hit trajectories): {avg_e_hit:.2f}")
    else:
        print("  Avg energy to t_hit using B@u (goal-hit trajectories): n/a (no goal-hit trajectories)")
    print()
    if n_success > 0:
        print("T_hit statistics (seconds):")
        print(f"  mean={reach_times.mean():.2f}  std={reach_times.std():.2f}")
        print(f"  p50={np.percentile(reach_times,50):.2f}  "
              f"p90={np.percentile(reach_times,90):.2f}  "
              f"p95={np.percentile(reach_times,95):.2f}  "
              f"p99={np.percentile(reach_times,99):.2f}  "
              f"max={reach_times.max():.2f}")
        print()
        print("Control energy  E = ∫ ||B u||² dt  (goal-hit trajectories):")
        print(f"  mean={energies.mean():.2f}  std={energies.std():.2f}")
        print(f"  p50={np.percentile(energies,50):.2f}  "
              f"p90={np.percentile(energies,90):.2f}  "
              f"p95={np.percentile(energies,95):.2f}  "
              f"p99={np.percentile(energies,99):.2f}  "
              f"max={energies.max():.2f}")
        print()
        print("Suggested energy_max for main.py  (bang-bang is worst case;")
        print("a trained smooth policy will need less energy):")
        print(f"  Tight    (p90): {np.percentile(energies, 90):.1f}")
        print(f"  Nominal  (p95): {np.percentile(energies, 95):.1f}")
        print(f"  Generous (p99): {np.percentile(energies, 99):.1f}")

    if need_paths:
        animate_mc_results(
            paths=mc["paths"],
            controls=mc["controls"],
            energy_hist=mc["energy_hist"],
            statuses=statuses,
            hit_steps=mc["hit_steps"],
            ts=mc["ts"],
            skip=int(args.anim_skip),
            max_traj=int(args.anim_trajs),
            save_path=args.save_animation,
            show=(not args.no_show),
        )
