import torch.nn as nn
import torch

class LinearControlNN(nn.Module):
    def __init__(self, prior_knowledge=True):
        super().__init__()
        # Standard fully-connected layer: 2 inputs -> 2 outputs, no bias
        self.fc = nn.Linear(2, 2, bias=False)

        if(prior_knowledge):
            #Initialize as diag(-1, -1)
            with torch.no_grad():
                self.fc.weight.copy_(torch.diag(torch.tensor([-1.0, -1.0])))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)