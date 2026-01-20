"""Test that all V networks use sigmoid activation."""
import torch
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.network import create_V
from src.hyperparameters import NetworkConfig


def test_v_network_uses_sigmoid():
    """Verify V network uses sigmoid activation."""
    net_cfg = NetworkConfig(n_inputs=2)
    V_net = create_V(net_cfg)

    assert V_net.activation_fn == torch.sigmoid, \
        f"V network must use sigmoid activation, got {V_net.activation_fn}"


if __name__ == "__main__":
    test_v_network_uses_sigmoid()
    print("Activation function test passed!")
