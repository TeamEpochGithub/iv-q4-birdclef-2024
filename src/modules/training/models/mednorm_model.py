import torch
from torch import nn


class MedNormModel(nn.Module):
    def __init__(self, model: nn.Module, use_vertical: bool = True, quantile: float = 0.25):
        super().__init__()
        self.model = model
        self.use_vertical = use_vertical
        self.quantile = quantile

    def forward(self, x):
        """Subtract the median over the time axis, and if specified also the frequency axis.

        :param x: The input tensor of shape (B, C, H, W)
        :return: The output tensor of shape (B, C, H, W)
        """
        norm = torch.quantile(x, self.quantile, dim=2, keepdim=True)[0]
        x = x - norm
        if self.use_vertical:
            norm = torch.quantile(x, self.quantile, dim=3, keepdim=True)[0]
            x = x - norm
        return self.model(x)

    def __repr__(self):
        return f"MedNormModel(model={self.model}, use_vertical={self.use_vertical}, quantile={self.quantile})"
