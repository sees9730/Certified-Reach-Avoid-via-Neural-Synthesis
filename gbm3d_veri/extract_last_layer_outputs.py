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
from src.dynamics import Dynamics, ClosedLoopDrift
from src.control_network import LinearControlNN
from src.save_load_utils import load_eval_bundle


torch.set_default_dtype(torch.float32)


# =============================================================================
# Build model + dynamics
# =============================================================================
def build_V_from_bundle(params: Hyperparameters, bundle_path: Path, device: str = "cpu", pretrain=False):
    V_net = create_V(params.network).to(device)
    if(pretrain):
        V_net.load_state_dict(torch.load(OUTPUT_DIR / "V_pretrained.pth", map_location=device))
    else:
        bundle = load_eval_bundle(bundle_path, map_location=device)
        V_net.load_state_dict(bundle["V_state_dict"])
    V_net.eval()
    return V_net

def build_dynamics(params: Hyperparameters, bundle_path: Path, device: str = "cpu", pretrain=False):
    u_nn = LinearControlNN(prior_knowledge=True, input_dim=params.network.n_inputs)
    if(pretrain):
        u_nn.load_state_dict(torch.load(OUTPUT_DIR / "controller_pretrained.pth", map_location=device))
    else:
        bundle = load_eval_bundle(bundle_path, map_location=device)
        u_nn.load_state_dict(bundle["control_state_dict"])

    def f_ol(x: torch.Tensor, u: torch.Tensor | None = None) -> torch.Tensor:
        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
        f1 = -1.5 * x1 + 1.0 * x2 + 0.0 * x3
        f2 = -1.0 * x1 - 1.5 * x2 + 1.0 * x3
        f3 =  0.0 * x1 - 1.0 * x2 - 1.5 * x3
        return torch.stack([f1, f2, f3], dim=1)

    g_coeffs = torch.tensor([0.2, 0.2, 0.2], dtype=torch.float32)

    def g(x: torch.Tensor) -> torch.Tensor:
        return g_coeffs.to(device=x.device, dtype=x.dtype) * x  # (N,3) diagonal diffusion entries

    f_cl = ClosedLoopDrift(f_ol, u_nn).to(params.training.device)
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
    params = Hyperparameters.default()
    params.network.n_inputs = 3
    params.network.n_hidden_1 = 64
    params.network.n_hidden_2 = 64
    params.network.n_outputs = 1
    params.network.input_scale = [100.0, 100.0, 100.0]
    params.network.scale_factor = 20.0

    init_range = np.array([[45.0, 55.0],
                           [-55.0, -45.0],
                           [50.0, 60.0]], dtype=np.float32)
    goal_range = np.array([[-25.0, 25.0],
                           [-25.0, 25.0],
                           [-25.0, 25.0]], dtype=np.float32)
    unsafe_range = np.array([[-100.0, -80.0],
                             [-100.0, 100.0],
                             [-100.0, -80.0]], dtype=np.float32)
    full_range = np.array([[-100.0, 100.0],
                           [-100.0, 100.0],
                           [-100.0, 100.0]], dtype=np.float32)
    
    ### Just need to Change these two parameters ###
    N_loop = 5
    N_samples = 5000 # N_samples per loop
    ################################################

    # models from bound training
    V_net = build_V_from_bundle(params, OUTPUT_DIR / "eval_bundle.pth", device="cpu")
    dynamics = build_dynamics(params, OUTPUT_DIR / "eval_bundle.pth", device="cpu")
    generate_scenario_data(N_loop, N_samples,
        full_range, init_range, unsafe_range, goal_range, V_net, dynamics,
        sample_features_path="samples_features.csv", x_full_path="x_full.csv")
    
    # NOTE: we did not save pre-training for this example
    # # models from pre-training
    # V_net = build_V_from_bundle(params, OUTPUT_DIR / "eval_bundle.pth", device="cpu", pretrain=True)
    # dynamics = build_dynamics(params, OUTPUT_DIR / "eval_bundle.pth", device="cpu", pretrain=True)
    # generate_scenario_data(N_loop, N_samples,
    #     full_range, init_range, unsafe_range, goal_range, V_net, dynamics,
    #     sample_features_path="samples_features_pretrain.csv", x_full_path="x_full_pretrain.csv")

    # Ignore below codes
    # w_last = V_net.output.weight[0] # extract the weights of the last layer
    # b_last = V_net.output.bias[0] # extract the bias of the last layer
    # # =============================================================================
    # # Example usage for scenario optimization
    # # =============================================================================
    # # Example 1: evalulate V(x) of the last layers on x_full samples
    # last_v_full = V_last_hidden(V_net, x_full) # shape (number of samples, last layer neurons) V(x full) >= 0
    # # Example 1.1: evaluate V(x) of the last layers on x_unsafe samples
    # last_v_unsafe = V_last_hidden(V_net, x_unsafe) # shape (number of unsafe samples, last layer neurons) V(x unsafe) >= 20
    # # Example 1.2: evaluate V(x) of the last layers on x_init samples
    # last_v_init = V_last_hidden(V_net, x_init) # shape (number of init samples, last layer neurons) V(x init) <= 1
    # # Example 2: evaluate GV(x) of the last layers on the generator samples
    # last_gv_gen = phi_features(V_net, x_gen, dynamics=dynamics) # shape (number of gen samples, last layer neurons) GV(x gen) < 0


if __name__ == "__main__":
    main()