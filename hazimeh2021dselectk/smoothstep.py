import torch
import torch.nn as nn

class SmoothStep(nn.Module):
    """
    Piecewise smooth step gating function as described in dselectk.
    """
    def __init__(self, gamma=0.5):
        """
        gamma (float): controls the width of the smooth transition region.
        """
        super(SmoothStep, self).__init__()
        self.gamma = gamma

def forward(self, x):
    gamma = self.gamma
    t = torch.clamp((x + gamma / 2) / gamma, 0, 1)
    out = torch.where(
        x < -gamma / 2,
        torch.zeros_like(x),
        torch.where(
            x > gamma / 2,
            torch.ones_like(x),
            -2 * (t ** 3) + 3 * t ** 2
        )
    )
    return out
