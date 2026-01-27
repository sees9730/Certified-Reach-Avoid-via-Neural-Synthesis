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
    sample_weighted_regions,
    V_last_hidden,
    phi_features,
    verify_G_decomposition,
    verify_V_decomposition,
    write_describe_dict_to_csv,
    save_x_full_to_csv
)
from src.hyperparameters import Hyperparameters
from src.network import create_V
from src.control_network import ZeroControl
from src.dynamics import Dynamics, ClosedLoopDrift
from src.save_load_utils import load_eval_bundle

torch.set_default_dtype(torch.float32)

# =============================================================================
# Build model + dynamics
# =============================================================================
def build_V(params: Hyperparameters, bundle_path: Path, device: str = "cpu", pretrain=False, model_seed=None):
    V_net = create_V(params.network).to(device)
    if(pretrain):
        ckpt_path = OUTPUT_DIR / "10D_GBM_pretrained_V" / f"V_pretrained_seed_{model_seed}.pth"
        V_net.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        bundle = load_eval_bundle(bundle_path, map_location=device)
        V_net.load_state_dict(bundle["V_state_dict"])
    V_net.eval()
    return V_net

def build_dynamics(params: Hyperparameters, bundle_path: Path, device: str = "cpu", pretrain=False):
    u_nn = ZeroControl(input_dim=params.network.n_inputs)

    def f_ol(x: torch.Tensor, u: torch.Tensor | None = None) -> torch.Tensor:
        x1  = x[:, 0]
        x2  = x[:, 1]
        x3  = x[:, 2]
        x4  = x[:, 3]
        x5  = x[:, 4]
        x6  = x[:, 5]
        x7  = x[:, 6]
        x8  = x[:, 7]
        x9  = x[:, 8]
        x10 = x[:, 9]

        f1  = -1.5 * x1  + 1.0 * x2
        f2  = -1.0 * x1  - 1.5 * x2  + 1.0 * x3
        f3  = -1.0 * x2  - 1.5 * x3  + 1.0 * x4
        f4  = -1.0 * x3  - 1.5 * x4  + 1.0 * x5
        f5  = -1.0 * x4  - 1.5 * x5  + 1.0 * x6
        f6  = -1.0 * x5  - 1.5 * x6  + 1.0 * x7
        f7  = -1.0 * x6  - 1.5 * x7  + 1.0 * x8
        f8  = -1.0 * x7  - 1.5 * x8  + 1.0 * x9
        f9  = -1.0 * x8  - 1.5 * x9  + 1.0 * x10
        f10 = -1.0 * x9  - 1.5 * x10

        return torch.stack([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10], dim=1)

    g_coeffs = torch.full((params.network.n_inputs,), 0.2, dtype=torch.float32)

    def g(x: torch.Tensor) -> torch.Tensor:
        return g_coeffs.to(device=x.device, dtype=x.dtype) * x  # (N,3) diagonal diffusion entries

    f_cl = ClosedLoopDrift(f_ol, u_nn).to(params.training.device)
    return Dynamics.dynamics(f=f_cl, g=g)


def generate_scenario_data(N_loop, N_samples, full_range, init_range, unsafe_range, goal_range, V_net, dynamics,
                           Use_weighted_uniform, weights_sample, sample_features_path, x_full_path):
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
        if(Use_weighted_uniform):
            x_full = sample_weighted_regions(
                N_samples, full_range, init_range, goal_range, unsafe_range,
                w_init=weights_sample[0], w_unsafe=weights_sample[1], w_goal=weights_sample[2], w_gen=weights_sample[3]
            )
        else:
            x_full, _, _, _ = sample_and_partition(
                N_samples=N_samples,
                full_range=full_range,
                init_range=init_range,
                goal_range=goal_range,
                unsafe_range=unsafe_range,
                device="cpu",
                dtype=torch.float32,
                seed=i
            )

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
    params.network.n_inputs = 10
    params.network.n_hidden_1 = 64
    params.network.n_hidden_2 = 64
    params.network.input_scale = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    params.network.scale_factor = 20.0
    init_range = np.array([[45.0, 55.0], [-55.0, -45.0], [45.0, 55.0], [45.0, 55.0], [45.0, 55.0], 
                           [45.0, 55.0], [45.0, 55.0], [45.0, 55.0], [45.0, 55.0], [45.0, 55.0]], dtype=np.float32)
    goal_range = np.array([[-25.0, 25.0], [-25.0, 25.0], [-25.0, 25.0], [-25.0, 25.0], [-25.0, 25.0], 
                           [-25.0, 25.0], [-25.0, 25.0], [-25.0, 25.0], [-25.0, 25.0], [-25.0, 25.0]], dtype=np.float32)
    unsafe_range = np.array([[-100.0, -80.0], [-100.0, 100.0], [-100.0, -80.0], [-100.0, -80.0], [-100.0, -80.0], 
                             [-100.0, -80.0], [-100.0, -80.0], [-100.0, -80.0], [-100.0, -80.0], [-100.0, -80.0]], dtype=np.float32)
    full_range = np.array([[-100.0, 100.0], [-100.0, 100.0], [-100.0, 100.0], [-100.0, 100.0], [-100.0, 100.0], 
                           [-100.0, 100.0], [-100.0, 100.0], [-100.0, 100.0], [-100.0, 100.0], [-100.0, 100.0]], dtype=np.float32)
    
    ### Just need to Change these five parameters ###
    N_loop = 5
    N_samples = 100 # N_samples per loop
    # If Use_weighted_uniform = False, then we do uniform sampling over full range
    # Else, do a weighted combination of uniform sampling over [init, unsafe, goal, generator] ranges
    Use_weighted_uniform = True 
    default_weights = [0.1, 0.1, 0.1, 0.7]
    ################################################

    # models from sample pre-training
    for i in range(5):
        model_seed = i
        V_net = build_V(params, OUTPUT_DIR / "eval_bundle.pth", device="cpu", pretrain=True, model_seed=model_seed)
        dynamics = build_dynamics(params, OUTPUT_DIR / "eval_bundle.pth", device="cpu", pretrain=True)
        generate_scenario_data(N_loop, N_samples,
            full_range, init_range, unsafe_range, goal_range, V_net, dynamics,
            Use_weighted_uniform, default_weights,
            sample_features_path="samples_features_pretrainmodel_"+str(model_seed)+".csv", 
            x_full_path="x_full_pretrainmodel_"+str(model_seed)+".csv")
    
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