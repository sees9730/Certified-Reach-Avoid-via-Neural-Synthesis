"""Test GV analytical implementation against autograd."""
import torch
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.phi_module import GV, GV_offset, verify_GV, create_GV
from src.dynamics import Dynamics
from src.network import create_V
from src.hyperparameters import NetworkConfig


class DiagonalDiffusion:
    """Diagonal diffusion: g(x) = 0.2 * x"""
    def __call__(self, x):
        return 0.2 * x


def test_GV_2d_diagonal_diffusion():
    """Test 2D GBM with diagonal diffusion."""
    net_cfg = NetworkConfig(n_inputs=2, input_scale=[100.0, 100.0])
    A = np.array([[-0.5, 1.0], [-1.0, -0.5]], dtype=np.float32)
    dynamics = Dynamics(f=A, g=DiagonalDiffusion())
    V_net = create_V(net_cfg)
    phi = GV(V_net, dynamics, scale_factor=1.0)

    assert verify_GV(phi, dynamics, n_samples=10000, tol=1e-6), \
        f"GV verification failed for 2D GBM with diagonal diffusion"


def test_GV_4d_chain_diffusion():
    """Test 4D chain dynamics."""
    net_cfg = NetworkConfig(n_inputs=4, input_scale=[100.0] * 4)
    A = np.array([
        [-1.5,  1.0,  0.0,  0.0],
        [-1.0, -1.5,  1.0,  0.0],
        [ 0.0, -1.0, -1.5,  1.0],
        [ 0.0,  0.0, -1.0, -1.5]
    ], dtype=np.float32)
    dynamics = Dynamics(f=A, g=DiagonalDiffusion())
    V_net = create_V(net_cfg)
    phi = GV(V_net, dynamics, scale_factor=1.0)

    assert verify_GV(phi, dynamics, n_samples=10000, tol=1e-6), \
        f"GV verification failed for 4D chain diffusion"


def test_GV_offset_2d_diagonal_diffusion():
    """Test GV_offset built via create_V/create_GV with input and output offsets."""
    net_cfg = NetworkConfig(n_inputs=2, input_scale=[100.0, 100.0])
    A = np.array([[-0.5, 1.0], [-1.0, -0.5]], dtype=np.float32)
    dynamics = Dynamics(f=A, g=DiagonalDiffusion())

    input_offset = [0.1, 0.2]
    output_offset = np.float32(0.1)
    V_net = create_V(net_cfg, input_offset=input_offset, output_offset=output_offset)
    phi = create_GV(
        V_net=V_net,
        dynamics=dynamics,
        network_config=net_cfg,
        input_offset=input_offset,
        include_time=False,
        include_energy=False,
    )

    assert isinstance(phi, GV_offset)
    assert torch.allclose(phi.input_offset, torch.tensor(input_offset, dtype=torch.float32)), \
        "GV_offset did not register the provided input_offset"
    assert verify_GV(phi, dynamics, n_samples=10000, tol=1e-6), \
        f"GV_offset verification failed for 2D GBM with input/output offsets"


if __name__ == "__main__":
    test_GV_2d_diagonal_diffusion()
    test_GV_4d_chain_diffusion()
    test_GV_offset_2d_diagonal_diffusion()
