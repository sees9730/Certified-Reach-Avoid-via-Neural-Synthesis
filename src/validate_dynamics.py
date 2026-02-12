from __future__ import annotations

from typing import Callable, Optional, Tuple, Dict, Any, Union
import numpy as np
import torch
from torch import nn


@torch.no_grad()
def estimate_reach_avoid_mc(
    f_cl_model: nn.Module,
    g_fn: Callable[[torch.Tensor], torch.Tensor],
    full_range: Union[np.ndarray, torch.Tensor],
    init_range: Union[np.ndarray, torch.Tensor],
    goal_range: Union[np.ndarray, torch.Tensor],
    unsafe_range: Optional[Union[np.ndarray, torch.Tensor]],
    dt: float,
    T: float,
    N_init: int,
    N_sim: int,
    seed: Optional[int] = None,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
    return_details: bool = False,
    anim: bool = True,
    N_traj: int = 10,   # <-- NEW
) -> Union[float, Tuple[float, Dict[str, Any]]]:
    """
    Monte-Carlo reach--avoid probability for Ito SDE using Euler-Maruyama.

    (docstring omitted for brevity)
    """
    # -----------------------
    # Convert ranges to torch
    # -----------------------
    dev = torch.device(device)

    def to_box(a) -> torch.Tensor:
        if isinstance(a, torch.Tensor):
            return a.to(device=dev, dtype=dtype)
        return torch.as_tensor(np.asarray(a), device=dev, dtype=dtype)

    full_box = to_box(full_range)
    init_box = to_box(init_range)
    goal_box = to_box(goal_range)

    assert full_box.ndim == 2 and full_box.shape[1] == 2, f"full_range must be (D,2), got {tuple(full_box.shape)}"
    D = int(full_box.shape[0])
    assert init_box.shape == (D, 2), f"init_range must be (D,2), got {tuple(init_box.shape)}"
    assert goal_box.shape == (D, 2), f"goal_range must be (D,2), got {tuple(goal_box.shape)}"
    assert dt > 0.0 and T > 0.0
    assert N_init >= 1 and N_sim >= 1

    # unsafe boxes
    unsafe_boxes = None
    K = 0
    if unsafe_range is not None:
        ub = to_box(unsafe_range)
        if ub.numel() > 0:
            assert ub.ndim == 2 and ub.shape[1] == 2, f"unsafe_range must be (*,2), got {tuple(ub.shape)}"
            assert ub.shape[0] % D == 0, f"unsafe_range first dim must be multiple of D={D}, got {ub.shape[0]}"
            K = int(ub.shape[0] // D)
            unsafe_boxes = ub.reshape(K, D, 2)

    # time discretization (hit exactly T)
    steps = int(np.ceil(T / dt))
    dt_eff = float(T / steps)
    sqrt_dt = float(np.sqrt(dt_eff))

    # RNG
    gen = torch.Generator(device=dev)
    if seed is None:
        seed = int(np.random.SeedSequence().entropy)
    gen.manual_seed(int(seed))

    # -----------------------
    # Box membership helpers
    # -----------------------
    def in_box(x: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        lo = box[:, 0]
        hi = box[:, 1]
        return torch.logical_and(x >= lo, x <= hi).all(dim=1)

    def in_any_unsafe(x: torch.Tensor) -> torch.Tensor:
        if unsafe_boxes is None:
            return torch.zeros((x.shape[0],), device=x.device, dtype=torch.bool)
        hit = torch.zeros((x.shape[0],), device=x.device, dtype=torch.bool)
        for k in range(K):
            hit |= in_box(x, unsafe_boxes[k])
        return hit

    # -----------------------
    # Sample initial states
    # -----------------------
    lo0 = init_box[:, 0]
    hi0 = init_box[:, 1]
    x0s = lo0 + (hi0 - lo0) * torch.rand((N_init, D), generator=gen, device=dev, dtype=dtype)

    # Ensure model on device/dtype
    f_cl_model = f_cl_model.to(device=dev, dtype=dtype)
    f_cl_model.eval()

    # -----------------------
    # (NEW) Prepare trajectory recording for animation
    # -----------------------
    traj_ids = None
    traj_store = None
    if anim:
        n_traj = int(max(1, min(N_traj, N_sim)))
        gen_sel = torch.Generator(device=dev)
        gen_sel.manual_seed(int(seed) + 1)  # separate RNG: does NOT affect MC
        traj_ids = torch.randperm(N_sim, generator=gen_sel, device=dev)[:n_traj]
        traj_store = torch.empty((steps + 1, n_traj, D), device=dev, dtype=dtype)

    # -----------------------
    # Main MC estimation
    # -----------------------
    per_init_p = torch.empty((N_init,), device=dev, dtype=torch.float32)
    per_init_success = torch.zeros((N_init,), device=dev, dtype=torch.int64)

    for i in range(N_init):
        x = x0s[i].unsqueeze(0).expand(N_sim, D).clone()

        # record initial states for the first batch
        if anim and traj_store is not None and i == 0:
            traj_store[0] = x[traj_ids]

        alive = torch.ones((N_sim,), device=dev, dtype=torch.bool)
        success = torch.zeros((N_sim,), device=dev, dtype=torch.bool)

        out_full = ~in_box(x, full_box)
        hit_unsafe = in_any_unsafe(x)
        hit_goal = in_box(x, goal_box)

        fail0 = out_full | hit_unsafe
        succ0 = (~fail0) & hit_goal

        alive &= ~(fail0 | succ0)
        success |= succ0

        for t in range(steps):  # <-- changed from "_" to "t" so we can store
            if not alive.any():
                # still store for animation (trajectory stays at last state)
                if anim and traj_store is not None and i == 0:
                    traj_store[t + 1] = x[traj_ids]
                break

            xa = x[alive]

            fa = f_cl_model(xa)
            if fa.shape != xa.shape:
                raise ValueError(f"f_cl_model(x) must return shape {tuple(xa.shape)}, got {tuple(fa.shape)}")

            ga = g_fn(xa)
            if not isinstance(ga, torch.Tensor):
                ga = torch.as_tensor(ga, device=dev, dtype=dtype)
            if ga.shape != xa.shape:
                raise ValueError(f"g_fn(x) must return shape {tuple(xa.shape)}, got {tuple(ga.shape)}")

            z = torch.randn((xa.shape[0], D), generator=gen, device=dev, dtype=dtype)
            xa_next = xa + fa * dt_eff + ga * (sqrt_dt * z)

            x[alive] = xa_next

            out_full = ~in_box(xa_next, full_box)
            hit_unsafe = in_any_unsafe(xa_next)
            hit_goal = in_box(xa_next, goal_box)

            fail = out_full | hit_unsafe
            succ = (~fail) & hit_goal

            idx_alive = alive.nonzero(as_tuple=False).squeeze(1)
            alive[idx_alive[fail]] = False
            alive[idx_alive[succ]] = False
            success[idx_alive[succ]] = True

            # record after update for the first batch
            if anim and traj_store is not None and i == 0:
                traj_store[t + 1] = x[traj_ids]

        successes = int(success.sum().item())
        per_init_success[i] = successes
        per_init_p[i] = float(successes) / float(N_sim)

    # -----------------------
    # (NEW) Plot the recorded N_traj trajectories from the actual MC run
    # -----------------------
    if anim and traj_store is not None:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        def phase_pairs(dim: int):
            if dim <= 1:
                return []
            pairs = [(0, 1)]
            if dim % 2 == 0:
                for j in range(2, dim, 2):
                    pairs.append((j, j + 1))
            else:
                for j in range(2, dim - 1, 2):
                    pairs.append((j, j + 1))
                pairs.append((dim - 2, dim - 1))  # overlap last pair
            return pairs

        def add_box_2d(ax, box_color, box: torch.Tensor, i: int, j: int, *, label: str, linestyle: str):
            xlo, xhi = float(box[i, 0].item()), float(box[i, 1].item())
            ylo, yhi = float(box[j, 0].item()), float(box[j, 1].item())
            rect = Rectangle(
                (xlo, ylo),
                (xhi - xlo),
                (yhi - ylo),
                fill=False,
                linewidth=2.0,
                edgecolor=box_color,
                linestyle=linestyle,
                label=label,
            )
            ax.add_patch(rect)

        pairs = phase_pairs(D)
        if len(pairs) > 0:
            n = len(pairs)
            fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 3.8), squeeze=False)

            traj_np = traj_store.detach().cpu().numpy()  # (steps+1, n_traj, D)
            n_traj = traj_np.shape[1]

            for ax, (a, b) in zip(axes[0], pairs):
                # trajectories (label only the first to keep legend clean)
                for m in range(n_traj):
                    ax.plot(
                        traj_np[:, m, a],
                        traj_np[:, m, b],
                        linewidth=1.2,
                        label=("traj" if m == 0 else "_nolegend_"),
                    )

                # mark starts/ends (also only once)
                ax.scatter(traj_np[0, 0, a], traj_np[0, 0, b], marker="o", s=30, label="start")
                ax.scatter(traj_np[-1, 0, a], traj_np[-1, 0, b], marker="x", s=35, label="end")

                # init + goal projections
                add_box_2d(ax, "green", init_box, a, b, label="init", linestyle="-")
                add_box_2d(ax, "blue", goal_box, a, b, label="goal", linestyle="-")

                # unsafe projections (if any)
                if unsafe_boxes is not None:
                    for k in range(K):
                        add_box_2d(
                            ax,
                            "red",
                            unsafe_boxes[k],
                            a,
                            b,
                            label=("unsafe" if k == 0 else "_nolegend_"),
                            linestyle="--",
                        )

                ax.set_xlabel(f"x{a+1}")
                ax.set_ylabel(f"x{b+1}")
                ax.grid(True, alpha=0.3)

            handles, labels = axes[0, 0].get_legend_handles_labels()
            seen = set()
            uniq_h, uniq_l = [], []
            for h, l in zip(handles, labels):
                if l not in seen and l != "_nolegend_":
                    uniq_h.append(h)
                    uniq_l.append(l)
                    seen.add(l)
            axes[0, 0].legend(uniq_h, uniq_l, loc="best")

            fig.tight_layout()
            plt.show()

    p_hat = float(per_init_p.min().item())

    if not return_details:
        return p_hat

    details: Dict[str, Any] = dict(
        p_hat=p_hat,
        per_init_p=per_init_p.detach().cpu().numpy(),
        per_init_success=per_init_success.detach().cpu().numpy(),
        x0s=x0s.detach().cpu().numpy(),
        dt=dt_eff,
        T=float(T),
        steps=int(steps),
        N_init=int(N_init),
        N_sim=int(N_sim),
        seed=int(seed),
    )
    return p_hat, details
