import torch.nn as nn
import torch
from smoothstep import SmoothStep
from helper import per_example_selector

class PerExampleGate(nn.Module):
    def __init__(self, p, k, m, gamma=0.5):
        super().__init__()
        self.p = p
        self.k = k
        self.m = m

        self.G = nn.Linear(p, k)
        self.W = nn.Linear(p, m)

        self.S = SmoothStep(gamma=gamma)

    def forward(self, x):
        B = x.size(0)
        gates = self.W(x).view(B, self.k, self.m)     # (B, k, m)
        soft_gates = self.S(gates)                    # (B, k, m)
        selector_probs = per_example_selector(soft_gates)         # (B, k, 2^m)
        weights = torch.softmax(self.G(x), dim=1)     # (B, k)

        # Weighted mixture over k gates per example
        mixture = torch.bmm(weights.unsqueeze(1), selector_probs).squeeze(1)  # (B, 2^m)
        return mixture
