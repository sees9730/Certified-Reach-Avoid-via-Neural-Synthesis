from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"
sys.path.insert(0, str(ROOT))

from src.scenario_utils import describe_samples_rows
from src.scenario_utils import (
    sample_and_partition,
    V_last_hidden,
    phi_features,
    verify_G_decomposition,
    verify_V_decomposition,
    write_describe_dict_to_csv,
    save_x_full_to_csv
)
from src.hyperparameters import Hyperparameters
from src.network import create_V
from src.dynamics import Dynamics
from src.save_load_utils import load_eval_bundle

from main import pi, DEG, CartpoleControlNN, ClosedLoopCartPole


torch.set_default_dtype(torch.float32)


# =============================================================================
# Build model + dynamics
# =============================================================================
def build_V_from_bundle(params: Hyperparameters, bundle_path: Path, device: str = "cpu", pretrain=False):
    input_offset = [0.0, 0.0, 0.0, 0.0]
    output_offset = np.float32(0.1)
    V_net = create_V(params.network, input_offset=input_offset, output_offset=output_offset)
    if(pretrain):
        V_net.load_state_dict(torch.load(OUTPUT_DIR / "V_pretrained.pth", map_location=device))
    else:
        bundle = load_eval_bundle(bundle_path, map_location=device)
        V_net.load_state_dict(bundle["V_state_dict"])
    V_net.eval()
    return V_net

def build_dynamics(params: Hyperparameters, bundle_path: Path, device: str = "cpu", pretrain=False):
    u_nn = CartpoleControlNN()
    if(pretrain):
        u_nn.load_state_dict(torch.load(OUTPUT_DIR / "controller_pretrained.pth", map_location=device))
    else:
        bundle = load_eval_bundle(bundle_path, map_location=device)
        u_nn.load_state_dict(bundle["control_state_dict"])

    g_coeffs = torch.tensor([0.0, 0.05, 0.0, 0.05], dtype=torch.float32)

    def g(x: torch.Tensor) -> torch.Tensor:
        """
        Diffusion term g(x) that is constant: [1.0, 1.0, 1.0] for every x.

        If x has shape (D,), returns (D,).
        If x has shape (N, D), returns (N, D) with each row [1.0, 1.0, 1.0].
        """
        base = g_coeffs.to(device=x.device, dtype=x.dtype)

        if x.dim() == 1:
            # x is shape (D,)
            return base
        elif x.dim() == 2:
            # x is shape (N, D)
            N = x.shape[0]
            return base.unsqueeze(0).expand(N, -1)  # (N, D)
        else:
            raise ValueError(f"g(x) expects x of shape (D,) or (N, D), got {tuple(x.shape)}")

    f_cl = ClosedLoopCartPole(controller=u_nn).to(device)
    return Dynamics.dynamics(f=f_cl, g=g)


def generate_scenario_data(N_loop, N_samples, full_range, init_range, unsafe_range, goal_range, V_net, dynamics,
                           sample_features_path, x_full_path):
    ###########################
    # Verifying the code
    ###########################

    _x_full, _, _, _x_gen = sample_and_partition(
        N_samples=100,
        full_range=full_range,
        init_range=init_range,
        goal_range=goal_range,
        unsafe_range=unsafe_range,
        device="cpu",
        dtype=torch.float32,
        seed=0
    )
    # Verifying the extraction of last layer features (you can comment this out if you want)
    v_max, v_mean = verify_V_decomposition(V_net, _x_full)
    print(f"V decomposition: max|err|={v_max.item():.3e}, mean|err|={v_mean.item():.3e}")
    if _x_gen.shape[0] > 0:
        g_max, g_mean = verify_G_decomposition(V_net, _x_gen, dynamics)
        print(f"G decomposition (x_gen): max|err|={g_max.item():.3e}, mean|err|={g_mean.item():.3e}")
    else:
        print("G decomposition skipped: x_gen is empty for the chosen N_samples/ranges.")
    del _x_full, _x_gen # release memory for verification


    ###########################
    # Main usage for scenario
    ###########################
    for i in range(N_loop):
        x_full, x_init, x_unsafe, x_gen = sample_and_partition(
            N_samples=N_samples,
            full_range=full_range,
            init_range=init_range,
            goal_range=goal_range,
            unsafe_range=unsafe_range,
            device="cpu",
            dtype=torch.float32,
            seed=i
        )
        # print("Number of all samples: ", x_full.shape)
        # print("Number of init samples: ", x_init.shape)
        # print("Number of unsafe samples: ", x_unsafe.shape)
        # print("Number of generator samples: ", x_gen.shape)

        data = describe_samples_rows(
            x_full,
            V_net=V_net, dynamics=dynamics,
            init_range=init_range, unsafe_range=unsafe_range, goal_range=goal_range,
            as_dict=True,
        )
        print(x_full[0])

        # Write [region labels, V, GV] to file
        write_describe_dict_to_csv(data, sample_features_path)
        # Write full samples to file
        save_x_full_to_csv(x_full, x_full_path)

    print("successfully write feature to: ", sample_features_path)
    print("successfully write x samples to: ", x_full_path)


def main():
    out_dir = HERE
    for p in out_dir.glob("*.csv"):
        p.unlink(missing_ok=True)  # Python 3.8+: ignore if already gone

    params = Hyperparameters.default()
    params.network.n_inputs = 4
    params.network.n_hidden_1 = 64
    params.network.n_hidden_2 = 64
    params.network.n_outputs = 1
    params.network.input_scale = [2*pi, 20.0, 10.0, 20.0]
    params.network.scale_factor = 20.0

    full_range = np.array([
        [-2*pi, 2*pi],
        [-20.0, 20.0],
        [-10.0, 10.0],
        [-20.0, 20.0],
    ], dtype=np.float32)

    init_range = np.array([
        [pi-15*DEG, pi+15*DEG],
        [-0.1, 0.1],
        [-1.0, 1.0],
        [-0.1, 0.1],
    ], dtype=np.float32)

    goal_range = np.array([
        [-0.4*pi, 0.4*pi],
        [-2.0, 2.0],
        [-2.0, 2.0],
        [-1.0, 1.0],
    ], dtype=np.float32)

    unsafe_min_z = np.array([
        full_range[0, :],
        full_range[1, :],
        [full_range[2, 0], full_range[2, 0]+0.5],
        full_range[3, :],
    ], dtype=np.float32)

    unsafe_max_z = np.array([
        full_range[0, :],
        full_range[1, :],
        [full_range[2, 1]-0.5, full_range[2, 1]],
        full_range[3, :],
    ], dtype=np.float32)

    unsafe_min_theta = np.array([
        [full_range[0, 0], full_range[0, 0]+0.5*DEG],
        full_range[1, :],
        full_range[2, :],
        full_range[3, :],
    ], dtype=np.float32)

    unsafe_max_theta = np.array([
        [full_range[0, 1]-0.5*DEG, full_range[0, 1]],
        full_range[1, :],
        full_range[2, :],
        full_range[3, :],
    ], dtype=np.float32)

    unsafe_min_z_dot = np.array([
        full_range[0, :],
        full_range[1, :],
        full_range[2, :],
        [full_range[3, 0], full_range[3,0]+0.5],
    ], dtype=np.float32)

    unsafe_max_z_dot = np.array([
        full_range[0, :],
        full_range[1, :],
        full_range[2, :],
        [full_range[3, 1]-0.5, full_range[3, 1]],
    ], dtype=np.float32)

    unsafe_min_theta_dot = np.array([
        full_range[0, :],
        [full_range[1, 0], full_range[1, 0]+0.5],
        full_range[2, :],
        full_range[3, :],
    ], dtype=np.float32)

    unsafe_max_theta_dot = np.array([
        full_range[0, :],
        [full_range[1, 1]-0.5, full_range[1, 1]],
        full_range[2, :],
        full_range[3, :],
    ], dtype=np.float32)

    unsafe_range = np.vstack((unsafe_min_z, unsafe_max_z,
                              unsafe_min_theta, unsafe_max_theta,
                              unsafe_min_z_dot, unsafe_max_z_dot,
                              unsafe_min_theta_dot, unsafe_max_theta_dot
                             ))
    
    ### Just need to Change these two parameters ###
    N_loop = 5
    N_samples = 5000 # N_samples per loop
    ################################################

    # models from bound training
    # V_net = build_V_from_bundle(params, OUTPUT_DIR / "eval_bundle.pth", device="cpu")
    # dynamics = build_dynamics(params, OUTPUT_DIR / "eval_bundle.pth", device="cpu")
    # generate_scenario_data(N_loop, N_samples,
    #     full_range, init_range, unsafe_range, goal_range, V_net, dynamics,
    #     sample_features_path="samples_features.csv", x_full_path="x_full.csv")
    
    # models from pre-training
    V_net = build_V_from_bundle(params, OUTPUT_DIR / "eval_bundle.pth", device="cpu", pretrain=True)
    dynamics = build_dynamics(params, OUTPUT_DIR / "eval_bundle.pth", device="cpu", pretrain=True)
    generate_scenario_data(N_loop, N_samples,
        full_range, init_range, unsafe_range, goal_range, V_net, dynamics,
        sample_features_path="samples_features_pretrain.csv", x_full_path="x_full_pretrain.csv")




if __name__ == "__main__":
    main()