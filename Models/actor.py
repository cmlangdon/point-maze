import torch.nn as nn
import torch

class ActorNet(nn.Module):
    def __init__(self):
        super(ActorNet, self).__init__()

        self.mlp = nn.Sequential(nn.Linear(6, 8, bias=False),
                                 nn.Tanh(),
                                 nn.Linear(8, 16, bias=False),
                                 nn.Tanh(),
                                 nn.Linear(16, 32, bias=False),
                                 nn.Tanh(),
                                 nn.Linear(32, 64, bias=False),
                                 nn.Tanh(),
                                 nn.Linear(64, 32, bias=False),
                                 nn.Tanh(),
                                 nn.Linear(32, 16, bias=False),
                                 nn.Tanh(),
                                 nn.Linear(16, 8, bias=False),
                                 nn.Tanh(),
                                 nn.Linear(8, 2, bias=False))

    def forward(self, x):
        return self.mlp(x)