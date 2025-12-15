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