import torch.nn as nn
import torch
import torch.nn.functional as F

class LinearControlNN(nn.Module):
      def __init__(self, prior_knowledge=True, input_dim=2):
          super().__init__()
          # Standard fully-connected layer: input_dim inputs -> input_dim outputs, no bias
          self.fc = nn.Linear(input_dim, input_dim, bias=False)

          if prior_knowledge:
              # Initialize as diagonal matrix of zeros for any input_dim
              with torch.no_grad():
                  self.fc.weight.copy_(torch.diag(torch.zeros(input_dim)))

      def forward(self, x: torch.Tensor) -> torch.Tensor:
          return self.fc(x)

class TanhPolicy(nn.Sequential):
    def __init__(
        self,
        n_in: int = 2,
        n_out: int = 1,
        n_hidden: int = 64,
        device: torch.device | str = "cpu"
    ):
        super().__init__(
            nn.Linear(n_in, n_hidden, dtype=torch.float32, device=device),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden, dtype=torch.float32, device=device),
            nn.Tanh(),
            nn.Linear(n_hidden, n_out, dtype=torch.float32, device=device),
        )

class GBMControlNN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=8, output_dim=2):
        super().__init__()
        # Fully connected layers
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)   # input -> hidden
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=False)  # hidden -> output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Hidden layer with ReLU activation
        x = F.tanh(self.fc1(x))
        # Output layer (no activation here; add if you need e.g. tanh/sigmoid/softmax)
        x = self.fc2(x)
        return x

class InvertControlNN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=8, output_dim=1):
        super().__init__()
        # Fully connected layers
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = F.tanh(self.fc1(x))
        out = F.tanh(self.fc2(h1))
        return out

class TanhPolicy(nn.Sequential):
    """
    A policy with three layers and tanh activations.
    """

    def __init__(
        self,
        n_in: int = 2,
        n_out: int = 1,
        n_hidden: int = 64,
        device: torch.device | str = "cpu"
    ):
        super().__init__(
            nn.Linear(n_in, n_hidden, dtype=torch.float32, device=device),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden, dtype=torch.float32, device=device),
            nn.Tanh(),
            nn.Linear(n_hidden, n_out, dtype=torch.float32, device=device),
        )

class WrapperConterlNN(nn.Module):
    """
    Wraps a 1D policy network and returns a 2D control:
        u(x) = [0, self.M_mLsquare * policy(x)]
    """

    def __init__(self, policy_net: nn.Module):
        super().__init__()
        self.policy_net = policy_net
        self.M_mLsquare = 6/(0.15*0.5**2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, 2) or (batch, state_dim)
        returns: (N, 2) where [:, 0] = policy(x), [:, 1] = 0
        """
        u2 = self.policy_net(x)          # shape (N, 1)
        zeros = torch.zeros_like(u2)     # same shape as u1
        u = torch.cat([zeros, self.M_mLsquare*u2], dim=-1)  # (N, 2)
        return u

class SwapStateWrapper(nn.Module):
    """
    Wrap a policy that expects state=[angular_rate, angle]
    so it can be called with state=[angle, angular_rate].

    Works with input shape (2,) or (N,2).
    """
    def __init__(self, base_policy: nn.Module):
        super().__init__()
        self.base_policy = base_policy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x_swapped = x[[1, 0]]  # [angle, ang_rate] -> [ang_rate, angle]
        elif x.ndim == 2:
            x_swapped = x[:, [1, 0]]
        else:
            raise ValueError(f"Expected x.ndim in {{1,2}}, got {x.ndim}")

        return self.base_policy(x_swapped)

class NonlinearControlNN(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=3):
        super().__init__()
        # Fully connected layers
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)   # input -> hidden
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=False)  # hidden -> output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = F.tanh(self.fc1(x))
        out = (self.fc2(h1))
        return out

class LorentzLinearControlNN(nn.Module):
    """
    3D linear feedback u = K(x) x with a *base-offset* only on two weights:

      u1 = (-23.71 + W[0,0]) * x1 + (-18.49 + W[0,1]) * x2 + W[0,2] * x3
      u2 =  W[1,0] * x1 +  W[1,1] * x2 + W[1,2] * x3
      u3 =  W[2,0] * x1 +  W[2,1] * x2 + W[2,2] * x3

    Implemented as a standard nn.Linear(3,3,bias=False), but in forward we add a fixed
    3x3 base-mask to the weight so only (0,0) and (0,1) get offsets.
    """
    def __init__(
        self,
        prior_knowledge: bool = False,
        input_dim: int = 3,
        k11_base: float = -23.71,
        k12_base: float = -18.49,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.fc = nn.Linear(3, 3, bias=False, device=device, dtype=dtype)
        with torch.no_grad():
            self.fc.weight.zero_()

        # Base-mask added to weights at forward-time: only affects W_eff[0,0], W_eff[0,1]
        base = torch.zeros((3, 3), device=device, dtype=dtype)
        if prior_knowledge:
            base[0, 0] = k11_base
            base[0, 1] = k12_base
        self.register_buffer("W_base", base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W_eff = self.fc.weight + self.W_base  # grad flows to fc.weight
        return F.linear(x, W_eff, bias=None)

    @torch.no_grad()
    def get_effective_weight(self) -> torch.Tensor:
        """Return the effective 3x3 matrix (CPU) used in forward."""
        return (self.fc.weight + self.W_base).detach().cpu().clone()

class Veh4DLinearControlNN(nn.Module):
    """Learnable linear feedback controller u = x @ K^T."""
    def __init__(self, K=None):
        super().__init__()
        self.fc = nn.Linear(4, 2, bias=False)
        if K is not None:
            with torch.no_grad():
                self.fc.weight.copy_(torch.as_tensor(K, dtype=self.fc.weight.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

class Veh4DControlNN(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=8, output_dim=2):
        super().__init__()
        # Fully connected layers
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = F.tanh(self.fc1(x))
        out = F.tanh(self.fc2(h1))
        return out

class Wrapper4DConterlNN(nn.Module):
    """
    Wraps a 2D policy network and returns a 4D control:

        u(x) = [0, M_mLsquare * policy(x)[:, 0], 0, M_mLsquare * policy(x)[:, 1]]

    Assumption:
      - policy_net(x) returns shape (N, 2)
      - x is (N, state_dim)
    """

    def __init__(self, policy_net: nn.Module, U_max: float):
        super().__init__()
        self.policy_net = policy_net
        self.register_buffer("U_max", torch.tensor(U_max, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u2d = self.policy_net(x)
        zeros = torch.zeros((x.size(0), 1), dtype=x.dtype, device=x.device)
        u = torch.cat(
            [zeros, 
             zeros, 
             self.U_max * u2d[:, 0:1], 
             self.U_max * u2d[:, 1:2]],
            dim=-1,
        )  # (N, 4)
        return u