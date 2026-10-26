from pathlib import Path
import os

import torch
from torch import nn

class ResidualAdapter(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(128, 3584, bias=False)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc2.weight)

    def forward(self, z):
        return self.fc2(self.act(self.fc1(z)))
