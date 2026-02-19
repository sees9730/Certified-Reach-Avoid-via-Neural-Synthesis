"""
2D Inverted Pendulum Synthesis with constant-but-unknown drift parameters.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"
sys.path.insert(0, str(ROOT))

from src.control_network import InvertControlNN, WrapperConterlNN
from src.discretization import discretize_regions
from src.dynamics import Dynamics
from src.hyperparameters import Hyperparameters
from src.network import create_V
from src.phi_module import create_GV
from src.regions import Region, Regions
from src.save_load_utils import enable_terminal_logging, save_eval_bundle, load_eval_bundle
from src.set_values import ClosedLoopSetValuedDrift, InvertedPendulumSetDrift
from src.trainer_const import train_network_bounds_const_theta
from src.training_utils import evaluate_constraints, print_constraint_summary
from src.utils import cleanup_and_setup_directories
from src.visualization import create_summary_plots

torch.manual_seed(0)


def get_param_ranges():
    g_nom, L_nom, b_nom, m_nom = 9.81, 0.5, 0.1, 0.15
    g_unc, L_unc, b_unc, m_unc = 0.0, 0.1, 0.0, 0.0
    return {
        "g": (g_nom - g_unc, g_nom + g_unc),
        "L": (L_nom - L_unc, L_nom + L_unc),
        "b": (b_nom - b_unc, b_nom + b_unc),
        "m": (m_nom - m_unc, m_nom + m_unc),
    }


def build_setvalued_dynamics(controller, device, state_dim, param_ranges):
    f_ol_set = InvertedPendulumSetDrift(
        g_range=param_ranges["g"],
        L_range=param_ranges["L"],
        b_range=param_ranges["b"],
        m_range=param_ranges["m"],
    ).to(device)

    g_coeffs = torch.tensor([0.0, 0.2], dtype=torch.float32)

    def g(x: torch.Tensor) -> torch.Tensor:
        base = g_coeffs.to(device=x.device, dtype=x.dtype)
        if x.dim() == 1:
            return base
        if x.dim() == 2:
            return base.unsqueeze(0).expand(x.shape[0], -1)
        raise ValueError(f"g(x) expects x of shape (2,) or (N, 2), got {tuple(x.shape)}")

    f_cl_module = ClosedLoopSetValuedDrift(f_ol_set, controller).to(device)
    return Dynamics.dynamics(f=f_cl_module, g=g, state_dim=state_dim)


def pretrain_network_samples(
    model,
    x_goal_range,
    x_unsafe_range,
    x_init_range,
    x_range,
    params,
    GV_net=None,
    num_epochs=1000,
    lr=1e-4,
    device='cpu',
    control_net=None,
    n_each: int = 400,
    lambda_w = 0.1,
    save_v_path=None,
    save_control_path=None,
):
    """
    Pre-train V and GV networks using sampled points.
    """
    print("\n" + "="*20)
    print("Pre-training using samples")
    print("="*20)

    opt_params = list(model.parameters())
    if control_net is not None:
        opt_params += list(control_net.parameters())
    optimizer = torch.optim.Adam(opt_params, lr=lr)

    if GV_net is not None:
        print("GV pre-training enabled")
    else:
        print("GV pre-training disabled")

    best_loss = float('inf')
    best_model_state = None
    best_control_state = None

    # Convert ranges to torch once: each is (D,2)
    x_range_t = torch.as_tensor(x_range, dtype=torch.float32, device=device)
    goal_t = torch.as_tensor(x_goal_range, dtype=torch.float32, device=device)
    init_t = torch.as_tensor(x_init_range, dtype=torch.float32, device=device)

    D = int(x_range_t.shape[0])
    low  = x_range_t[:, 0]
    high = x_range_t[:, 1]
    span = high - low

    def _sample_in_box(box: torch.Tensor, N: int) -> torch.Tensor:
        b_low = box[:, 0]
        b_high = box[:, 1]
        return torch.rand(N, D, device=device) * (b_high - b_low) + b_low

    def _in_box(x_batch: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        return ((x_batch >= box[:, 0]) & (x_batch <= box[:, 1])).all(dim=1)

    def _unsafe_to_boxes(x_unsafe) -> torch.Tensor:
        t = torch.as_tensor(x_unsafe, dtype=torch.float32, device=device)

        if t.dim() == 2:
            if t.shape == (D, 2):
                return t.unsqueeze(0)  # (1,D,2)
            if t.shape[1] == 2 and (t.shape[0] % D == 0):
                K = int(t.shape[0] // D)
                return t.view(K, D, 2)  # (K,D,2)  <-- handles vstack case
            raise ValueError(f"x_unsafe_range 2D must be (D,2) or (K*D,2); got {tuple(t.shape)}")

        if t.dim() == 3:
            if t.shape[1:] != (D, 2):
                raise ValueError(f"x_unsafe_range 3D must be (K,D,2) with D={D}; got {tuple(t.shape)}")
            return t

        raise ValueError(f"x_unsafe_range must be (D,2), (K,D,2), or (K*D,2); got {tuple(t.shape)}")

    unsafe_boxes = _unsafe_to_boxes(x_unsafe_range)  # (K,D,2)
    K_unsafe = int(unsafe_boxes.shape[0])

    def _in_unsafe_union(x_batch: torch.Tensor) -> torch.Tensor:
        """mask True if x is inside ANY unsafe box."""
        mask = torch.zeros(x_batch.shape[0], dtype=torch.bool, device=device)
        for k in range(K_unsafe):
            mask |= _in_box(x_batch, unsafe_boxes[k])
        return mask

    def _sample_in_unsafe_union(N: int) -> torch.Tensor:
        if N <= 0:
            raise ValueError(f"N must be positive, got {N}")

        if K_unsafe == 1:
            return _sample_in_box(unsafe_boxes[0], N)

        xs = []
        for k in range(K_unsafe):
            xs.append(_sample_in_box(unsafe_boxes[k], N))  # (N, D) per box

        x = torch.cat(xs, dim=0)  # (K_unsafe * N, D)
        x = x[torch.randperm(x.shape[0], device=device)]  # shuffle
        return x
    
    def _l2_weight_penalty(model: torch.nn.Module, exclude_bias: bool = True) -> torch.Tensor:
        reg = 0.0
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if exclude_bias and (p.dim() == 1 or name.endswith("bias")):
                continue
            reg = reg + (p ** 2).sum()
        return reg

    # Main training loop
    for epoch in range(num_epochs):
        model.train()
        if control_net is not None:
            control_net.train()
        if GV_net is not None:
            GV_net.train()

        # full-range samples -> enforce v(x) >= 0
        x_full = torch.rand(n_each, D, device=device) * span + low
        v_full = model(x_full).squeeze(-1)
        v_loss_full = F.relu(0.0 - v_full).sum()

        # init-range samples -> enforce v(x) <= 1
        x_init = _sample_in_box(init_t, n_each)
        v_init = model(x_init).squeeze(-1)
        v_loss_init = F.relu(v_init - 1.0).sum()

        # unsafe-range samples -> enforce v(x) >= beta_ra
        x_unsafe = _sample_in_unsafe_union(int(n_each /6))
        v_unsafe = model(x_unsafe).squeeze(-1)
        v_loss_unsafe = F.relu(params.constraints.beta_ra - v_unsafe).sum()

        # samples inside full-range but outside (goal ∪ unsafe) -> enforce v(x) >= 0.0
        x_others_list = []
        need = n_each
        max_tries = 20
        tries = 0
        while need > 0 and tries < max_tries:
            tries += 1
            x_cand = torch.rand(max(need * 4, 32), D, device=device) * span + low
            cand_in_goal = _in_box(x_cand, goal_t)
            cand_in_unsafe = _in_unsafe_union(x_cand)
            keep = ~(cand_in_goal | cand_in_unsafe)
            x_keep = x_cand[keep]
            if x_keep.shape[0] > 0:
                take = min(need, x_keep.shape[0])
                x_others_list.append(x_keep[:take])
                need -= take

        if len(x_others_list) == 0:
            x_others = x_full.detach()
        else:
            x_others = torch.cat(x_others_list, dim=0)

        loss_v = (
            v_loss_full
            + v_loss_init
            + v_loss_unsafe
        )

        loss_gv = torch.tensor(0.0, device=device)
        if GV_net is not None and x_others.numel() > 0:
            x_gv = x_others.detach().clone().requires_grad_(True)
            gv_output = GV_net(x_gv).squeeze(-1)
            loss_gv = F.relu(gv_output).sum()

        total_loss = loss_v + loss_gv

        # L2 weight penalty
        reg_w = _l2_weight_penalty(model, exclude_bias=True)
        total_loss = total_loss + lambda_w * reg_w

        # Track best
        if total_loss.item() <= best_loss:
            best_loss = total_loss.item()
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if control_net is not None:
                best_control_state = {k: v.detach().cpu().clone() for k, v in control_net.state_dict().items()}

        if epoch % 100 == 0:
            if GV_net is not None:
                print(f"Epoch {epoch} | V_loss={loss_v.item():8.4f} | GV_loss={loss_gv.item():8.4f}")
            else:
                print(f"Epoch {epoch} | V_loss={loss_v.item():8.4f}")

        # Optimize/update networks
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if control_net is not None and best_control_state is not None:
            control_net.load_state_dict(best_control_state)
        print(f"\nBest loss: {best_loss:.6f}")

        # Save best pretrained weights (state_dict)
        if save_v_path is not None:
            torch.save(best_model_state, save_v_path)
            print(f"Saved pretrained V_net to: {save_v_path}")
        if (control_net is not None) and (best_control_state is not None) and (save_control_path is not None):
            torch.save(best_control_state, save_control_path)
            print(f"Saved pretrained Controller_net to: {save_control_path}")

    networks_trained = ["V"]
    if GV_net is not None:
        networks_trained.append("GV")
    if control_net is not None:
        networks_trained.append("Controller")
    print(f"Pre-training complete. {' + '.join(networks_trained)} initialized.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--train", type=int, default=1, choices=[0, 1])
    args = parser.parse_args()

    print("=" * 20)
    print("2D Inverted Pendulum (const-theta)")
    print("=" * 20)

    # === Hyperparameters ===
    params = Hyperparameters.default()

    params.network.n_inputs = 2
    params.network.n_hidden_1 = 64
    params.network.n_hidden_2 = 64
    pi = np.pi
    params.network.input_scale = [2*pi, 20.0]
    params.network.scale_factor = 20.0

    params.training.learning_rate = 0.005
    params.training.num_epochs = 100000
    params.training.generator_weight = 1.0
    params.training.generator_start_epoch = 0
    params.logging.visualize_interval = 0

    params.discretization.n_goal = 30
    params.discretization.n_outside_goal = 4
    params.discretization.n_generator = 4
    params.discretization.n_unsafe = 15
    params.discretization.n_init = 30

    params.constraints.beta_ra = 20.0

    params.compute_V = True
    params.compute_GV = True

    params.training.enable_pretraining = True
    params.training.pretrain_epochs = 5000
    params.training.pretrain_lr = 0.01
    params.training.pretrain_n_samples = 1200

    params.refinement.v_outside.enable_refinement = True
    params.refinement.v_outside.refine_interval = 500
    params.refinement.v_outside.late_epoch_threshold = 2500
    params.refinement.v_outside.refine_interval_late = 100
    params.refinement.v_outside.refine_factor = 2
    params.refinement.v_outside.max_cells = 50000
    params.refinement.v_outside.N_to_refine = 999999

    params.refinement.v_outside.enable_merging = True
    params.refinement.v_outside.merge_interval = 501
    params.refinement.v_outside.merge_max_passes = 8
    params.refinement.v_outside.merge_relax_margin = 0.5

    params.refinement.gv_generator.enable_refinement = True
    params.refinement.gv_generator.refine_interval = 500
    params.refinement.gv_generator.late_epoch_threshold = 3500
    params.refinement.gv_generator.refine_interval_late = 100
    params.refinement.gv_generator.refine_factor = 2
    params.refinement.gv_generator.max_cells = 50000
    params.refinement.gv_generator.N_to_refine = 999999

    params.refinement.gv_generator.enable_merging = True
    params.refinement.gv_generator.merge_interval = 501
    params.refinement.gv_generator.merge_max_passes = 8
    params.refinement.gv_generator.merge_relax_margin = -500.0

    param_ranges = get_param_ranges()
    theta_grid_splits = [1, 8, 1, 1]

    # Controller + dynamics
    rl_policy_net = InvertControlNN()
    u_nn = WrapperConterlNN(rl_policy_net)
    dynamics = build_setvalued_dynamics(
        controller=u_nn,
        device=params.training.device,
        state_dim=params.network.n_inputs,
        param_ranges=param_ranges,
    )

    # Regions
    init_range = np.array([[(3 / 4) * pi, (5 / 4) * pi], [-1.0, 1.0]], dtype=np.float32)
    goal_range = np.array([[-0.4 * pi, 0.4 * pi], [-4.0, 4.0]], dtype=np.float32)
    unsafe_down1 = np.array([[-2 * pi, -2 * pi + 0.5 * pi], [-20.0, -10.0]], dtype=np.float32)
    unsafe_down2 = np.array([[2 * pi - 0.5 * pi, 2 * pi], [10.0, 20.0]], dtype=np.float32)
    unsafe_lb = np.array([[-2 * pi, -2 * pi + 0.5], [-20.0, 20.0]], dtype=np.float32)
    unsafe_rb = np.array([[2 * pi - 0.5, 2 * pi], [-20.0, 20.0]], dtype=np.float32)
    unsafe_tb = np.array([[-2 * pi, 2 * pi], [20.0 - 0.5, 20.0]], dtype=np.float32)
    unsafe_bb = np.array([[-2 * pi, 2 * pi], [-20.0, -20.0 + 0.5]], dtype=np.float32)
    unsafe_range = np.vstack((unsafe_down1, unsafe_down2, unsafe_tb, unsafe_bb, unsafe_lb, unsafe_rb))
    full_range = np.array([[-2 * pi, 2 * pi], [-20.0, 20.0]], dtype=np.float32)

    init = Region(init_range)
    goal = Region(goal_range)
    unsafe = Region.union(
        Region(unsafe_down1), Region(unsafe_down2), Region(unsafe_tb),
        Region(unsafe_bb), Region(unsafe_lb), Region(unsafe_rb)
    )
    full = Region(full_range)
    regions = Regions(init=init, goal=goal, unsafe=unsafe, full=full)

    # Networks
    V_net = create_V(params.network)
    GV_net = create_GV(V_net=V_net, dynamics=dynamics, network_config=params.network)

    # Discretization
    region_cells = discretize_regions(regions, params.discretization, use_radial_generator=False)

    # === Setup ===
    device = params.training.device
    bundle_path = OUTPUT_DIR / "eval_bundle.pth"
    def evaluate_final_constraints(V_net, GV_net, region_cells, params):
        return evaluate_constraints(
            V_net=V_net,
            GV_net=GV_net,
            region_cells=region_cells,
            beta_ra=params.constraints.beta_ra,
            device=params.training.device,
            theta_ranges=param_ranges,
            theta_grid_splits=theta_grid_splits,
            adv_delta=1e-4,
        )

    if args.train == 1:
        cleanup_and_setup_directories(["results", "training_progress"])
        enable_terminal_logging(OUTPUT_DIR / "terminal_log.txt")
        start_time = time.time()

        if params.training.enable_pretraining:
            pretrain_network_samples(
                model=V_net,
                x_goal_range=goal_range,
                x_unsafe_range=unsafe_range,
                x_init_range=init_range,
                x_range=full_range,
                params=params,
                GV_net=GV_net,
                num_epochs=params.training.pretrain_epochs,
                lr=params.training.pretrain_lr,
                device=params.training.device,
                control_net=u_nn,
                n_each=params.training.pretrain_n_samples,
                save_v_path=OUTPUT_DIR / "V_pretrained.pth",
                save_control_path=OUTPUT_DIR / "controller_pretrained.pth"
            )

        def create_scheduler(optimizer):
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.95)

        loss_history, final_beta_s, refinement_epochs = train_network_bounds_const_theta(
            V_net=V_net,
            GV_net=GV_net,
            region_cells=region_cells,
            regions=regions,
            params=params,
            theta_ranges=param_ranges,
            theta_grid_splits=theta_grid_splits,
            control_net=u_nn,
            create_scheduler=create_scheduler,
            start_time=start_time,
            adv_delta=1e-4,
        )

        elapsed = time.time() - start_time
        print(f"Training complete in {elapsed:.2f}s")

        results = evaluate_final_constraints(V_net, GV_net, region_cells, params)
        print_constraint_summary(results)
        if results.get("theta_cell_worst_idx", -1) >= 0:
            lo = results["theta_cell_worst_lo"]
            hi = results["theta_cell_worst_hi"]
            print(
                f"theta_cell_worst[{results['theta_cell_worst_idx']}]: "
                f"g[{lo['g']:.4f},{hi['g']:.4f}] "
                f"L[{lo['L']:.4f},{hi['L']:.4f}] "
                f"b[{lo['b']:.4f},{hi['b']:.4f}] "
                f"m[{lo['m']:.4f},{hi['m']:.4f}] "
                f"obj={results['theta_cell_worst_obj']:.6f}"
            )

        save_eval_bundle(
            OUTPUT_DIR,
            V_net=V_net,
            GV_net=GV_net,
            control_net=u_nn,
            params=params,
            regions=regions,
            region_cells=region_cells,
            final_beta_s=final_beta_s,
            loss_history=loss_history,
            refinement_epochs=refinement_epochs,
            results=results,
        )

        create_summary_plots(
            V_net=V_net,
            GV_net=GV_net,
            regions=regions,
            region_cells=region_cells,
            beta_s=final_beta_s,
            beta_ra=params.constraints.beta_ra,
            loss_history=loss_history,
            refinement_epochs=refinement_epochs,
            results=results,
            output_dir="results",
        )
    else:
        print("\n" + "="*20)
        print("Loading Bundle")
        print("="*20)

        bundle = load_eval_bundle(bundle_path, map_location="cpu")

        params = Hyperparameters.from_dict(bundle["hyperparameters"])
        params.training.device = device
        region_cells = bundle["region_cells"]

        V_net = create_V(params.network).to(params.training.device)
        V_net.load_state_dict(bundle["V_state_dict"])

        rl_policy_net = InvertControlNN()
        u_nn = WrapperConterlNN(rl_policy_net).to(params.training.device)
        if bundle["control_state_dict"] is not None:
            u_nn.load_state_dict(bundle["control_state_dict"])

        dynamics = build_setvalued_dynamics(
            controller=u_nn,
            device=params.training.device,
            state_dim=params.network.n_inputs,
            param_ranges=param_ranges,
        )

        GV_net = create_GV(
            V_net=V_net,
            dynamics=dynamics,
            network_config=params.network,
        ).to(params.training.device)
        if bundle["GV_state_dict"] is not None:
            GV_net.load_state_dict(bundle["GV_state_dict"], strict=False)

        region_cells = {
            k: [(lo.to(params.training.device), hi.to(params.training.device)) for (lo, hi) in v]
            for k, v in region_cells.items()
        }

        final_beta_s = bundle["final_beta_s"]
        loss_history = bundle["loss_history"]
        refinement_epochs = bundle["refinement_epochs"]

        print("Recomputing final evaluation with theta-grid generator bounds...")
        results = evaluate_final_constraints(V_net, GV_net, region_cells, params)

        print("\n" + "="*20)
        print("Final Evaluation (loaded)")
        print("="*20)
        print_constraint_summary(results)
        if results.get("theta_cell_worst_idx", -1) >= 0:
            lo = results["theta_cell_worst_lo"]
            hi = results["theta_cell_worst_hi"]
            print(
                f"theta_cell_worst[{results['theta_cell_worst_idx']}]: "
                f"g[{lo['g']:.4f},{hi['g']:.4f}] "
                f"L[{lo['L']:.4f},{hi['L']:.4f}] "
                f"b[{lo['b']:.4f},{hi['b']:.4f}] "
                f"m[{lo['m']:.4f},{hi['m']:.4f}] "
                f"obj={results['theta_cell_worst_obj']:.6f}"
            )

        if final_beta_s is None:
            final_beta_s = float(results.get("V_outside_min", 0.0))

        create_summary_plots(
            V_net=V_net,
            GV_net=GV_net,
            regions=regions,
            region_cells=region_cells,
            beta_s=final_beta_s,
            beta_ra=params.constraints.beta_ra,
            loss_history=loss_history,
            refinement_epochs=refinement_epochs,
            results=results,
            output_dir="results",
        )


if __name__ == '__main__':
    # Quick check for benchmark mode before full arg parse
    import sys
    is_benchmark = '--benchmark=1' in sys.argv or '--benchmark' in sys.argv and '1' in sys.argv

    if is_benchmark:
        n_runs = 2
        times = []
        cells = []

        print("="*20)
        print(f"Benchmark mode: {n_runs} runs")
        print("="*20)

        for i in range(n_runs):
            print(f"\n*** Run {i+1}/{n_runs} ***")
            torch.manual_seed(i)
            training_time = main(benchmark_mode=True)
            if training_time is not None:
                times.append(training_time)

            # Extract cell counts from last run
            bundle_path = OUTPUT_DIR / "eval_bundle.pth"
            if bundle_path.exists():
                import torch
                bundle = torch.load(bundle_path, map_location='cpu')
                cell_counts = {
                    'init': len(bundle['region_cells']['init']),
                    'goal': len(bundle['region_cells']['goal']),
                    'unsafe': len(bundle['region_cells']['unsafe']),
                    'outside': len(bundle['region_cells']['outside']),
                    'generator': len(bundle['region_cells']['generator']),
                }
                cell_counts['total_v'] = cell_counts['init'] + cell_counts['goal'] + cell_counts['unsafe'] + cell_counts['outside']
                cell_counts['total'] = cell_counts['total_v'] + cell_counts['generator']
                cells.append(cell_counts)
    else:
        main()
