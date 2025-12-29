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
    def __init__(self, input_dim=2, hidden_dim=4, output_dim=2):
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


# class NonlinearControlNN(nn.Module):
#     def __init__(self, input_dim=3, hidden_dim=8, output_dim=3):
#         super().__init__()
#         # Fully connected layers
#         self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)   # input -> hidden
#         self.fc2 = nn.Linear(hidden_dim, output_dim, bias=False)  # hidden -> output

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         h1 = F.tanh(self.fc1(x))
#         out = F.tanh(self.fc2(h1))
#         return out

class NonlinearControlNN(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=8, output_dim=3):
        super().__init__()
        # Fully connected layers
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)    # input -> hidden1
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=True)   # hidden1 -> hidden2
        self.fc3 = nn.Linear(hidden_dim, hidden_dim, bias=True)  # hidden2 -> hidden3
        self.fc4 = nn.Linear(hidden_dim, hidden_dim, bias=True)  # hidden2 -> hidden3
        self.fc5 = nn.Linear(hidden_dim, hidden_dim, bias=True)  # hidden2 -> hidden3
        self.fc6 = nn.Linear(hidden_dim, hidden_dim, bias=True)  # hidden2 -> hidden3
        self.fc7 = nn.Linear(hidden_dim, output_dim, bias=False)  # hidden3 -> output


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = F.tanh(self.fc1(x))
        h2 = F.tanh(self.fc2(h1))
        h3 = F.tanh(self.fc3(h2))
        h4 = F.tanh(self.fc4(h3))
        h5 = F.tanh(self.fc5(h4))
        h6 = F.tanh(self.fc6(h5))
        out = F.tanh(self.fc7(h6))
        return out

# class NonlinearControlNN(nn.Module):
#     def __init__(self, input_dim=3, hidden_dim=8, output_dim=3):
#         super().__init__()
#         # Fully connected layers
#         self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)   # input -> hidden
#         self.fc2 = nn.Linear(hidden_dim, output_dim, bias=False)  # hidden -> output

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         h1 = F.sigmoid(self.fc1(x))
#         out = 10.0 * F.sigmoid(self.fc2(h1))
#         return out