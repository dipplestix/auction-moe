import torch.nn as nn
import torch
from smoothstep import SmoothStep
from helper import selector

class StaticGate(nn.Module):
    def __init__(self, k, m, gamma=0.5):
        super().__init__()
        self.k = k
        self.m = m

        self.alpha = nn.Parameter(torch.randn(k))
        self.Z = nn.Parameter(torch.randn(k, m))
        self.S = SmoothStep(gamma=gamma)

    def forward(self):
        out = self.S(self.Z)        # shape (k, m)
        probs = selector(out)       # shape (k, 2^m)
        weights = torch.softmax(self.alpha, dim=0)  # shape (k,)
        mixture = weights @ probs   # shape (2^m,)
        return mixture
