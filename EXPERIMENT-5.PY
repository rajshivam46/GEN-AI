import torch
import torch.nn as nn

class SimpleFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.shift = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        z = x * self.scale + self.shift
        log_det = torch.log(torch.abs(self.scale))
        return z, log_det

# Example
x = torch.randn(10)
flow = SimpleFlow()

z, log_det = flow(x)
print("Transformed:", z)
print("Log determinant:", log_det)
