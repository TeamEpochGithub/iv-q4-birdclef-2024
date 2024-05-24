import torch
from torch import nn


class AsymmetricalFocalLoss(nn.Module):
    def __init__(self, gamma=0, zeta=0):
        super(AsymmetricalFocalLoss, self).__init__()
        self.gamma = gamma   # balancing between classes
        self.zeta = zeta     # balancing between active/inactive frames

    def forward(self, pred, target):
        losses = - (((1 - pred) ** self.gamma) * target * torch.clamp_min(torch.log(pred), -100) +
                    (pred ** self.zeta) * (1 - target) * torch.clamp_min(torch.log(1 - pred), -100))
        return torch.mean(losses)