import torch.nn as nn
import torch
import torch.nn.functional as F


class LinearControlNN(nn.Module):
    def __init__(self, prior_knowledge=True, input_dim=2):
        super().__init__()
        # Standard fully-connected layer: 2 inputs -> 2 outputs, no bias
        self.fc = nn.Linear(input_dim, input_dim, bias=False)

        if(prior_knowledge):
            if(input_dim == 2):
                #Initialize as diag(-1, -1)
                with torch.no_grad():
                    self.fc.weight.copy_(torch.diag(torch.tensor([-1.0, -1.0])))
            if(input_dim == 3):
                #Initialize as diag(0, 0)
                with torch.no_grad():
                    self.fc.weight.copy_(torch.diag(torch.tensor([0.0, 0.0, 0.0])))
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
    

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
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)   # input -> hidden
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=False)  # hidden -> output

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
    

# class LorentzLinearControlNN(nn.Module):
#     def __init__(self, prior_knowledge: bool = True, input_dim: int = 3):
#         super().__init__()
#         if input_dim != 3:
#             raise ValueError(f"LinearControlNN here is specialized to input_dim=3, got {input_dim}")

#         # Fully-connected: 3 inputs -> 3 outputs, no bias
#         self.fc = nn.Linear(input_dim, input_dim, bias=False)

#         if prior_knowledge:
#             # Want: u1 = -23.71*x1 -18.49*x2 + 0*x3), u2 = 0, u3 = 0
#             # So set first row = [-23.71, -18.49, 0], other rows = [0,0,0]
#             W = torch.zeros((3, 3), dtype=torch.float32)
#             W[0, 0] = -23.71
#             W[0, 1] = -18.49
#             # W[0, 2] = 0.0 already
#             with torch.no_grad():
#                 self.fc.weight.copy_(W)
#         else:
#             # default: all zeros (you can change this if desired)
#             with torch.no_grad():
#                 self.fc.weight.zero_()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if x.shape[-1] != 3:
#             raise ValueError(f"Expected x last-dim = 3, got shape {tuple(x.shape)}")
#         return self.fc(x)
    

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
        prior_knowledge: bool = True,
        input_dim: int = 3,
        k11_base: float = -23.71,
        k12_base: float = -18.49,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if input_dim != 3:
            raise ValueError(f"LorentzLinearControlNN is specialized to input_dim=3, got {input_dim}")

        self.fc = nn.Linear(3, 3, bias=False, device=device, dtype=dtype)

        # Base-mask added to weights at forward-time: only affects W_eff[0,0], W_eff[0,1]
        base = torch.zeros((3, 3), device=device, dtype=dtype)
        base[0, 0] = k11_base
        base[0, 1] = k12_base
        self.register_buffer("W_base", base)

        if prior_knowledge:
            # Initialize *learnable* W to zeros so u1 starts at the base rule,
            # and u2,u3 start as 0 (you can change these if you want).
            with torch.no_grad():
                self.fc.weight.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != 3:
            raise ValueError(f"Expected x last-dim = 3, got shape {tuple(x.shape)}")

        W_eff = self.fc.weight + self.W_base  # grad flows to fc.weight
        return F.linear(x, W_eff, bias=None)

    @torch.no_grad()
    def get_effective_weight(self) -> torch.Tensor:
        """Return the effective 3x3 matrix (CPU) used in forward."""
        return (self.fc.weight + self.W_base).detach().cpu().clone()
