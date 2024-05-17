"""Normalizer for Spectrograms."""

import torch
from torch import nn


class SpecNormalize(nn.Module):
    """Normalization module for spectrograms.

    :param eps: The epsilon value to add to the denominator.
    """

    eps: float

    def __init__(self, eps: float = 1e-8) -> None:
        """Initialize the module.

        :param eps: The epsilon value to add to the denominator.
        """
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply minmax scaler to the data.

        :param x: The input data.
        :return: The normalized data.
        """
        # x: (batch, channel, freq, time)

        min_ = x.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        max_ = x.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]

        return (x - min_) / (max_ - min_ + self.eps)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Call forward method on log10 of input.

        :param x: The input data.
        :return: The normalized data.
        """
        # log10 returns some -inf's replace them with zeros
        return self.forward(torch.nan_to_num(torch.log10(x + self.eps), nan=0, neginf=-10, posinf=1))
