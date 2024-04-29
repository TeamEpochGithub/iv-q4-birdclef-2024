"""Focal loss implementation, for combating class imbalance in classification tasks."""

import torch
import torch.nn.functional as ff
from torch import nn


class FocalLoss(nn.Module):
    """Focal loss implementation, for combating class imbalance in classification tasks."""

    def __init__(self, alpha: float = 0.8, gamma: float = 2) -> None:
        """Initialize the focal loss.

        :param alpha: The alpha parameter.
        :param gamma: The gamma parameter.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the focal loss.

        :param inputs: The model predictions.
        :param targets: The true labels.

        :return: The focal loss.
        """
        # Flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        # First compute binary cross-entropy
        bce = ff.binary_cross_entropy(inputs, targets, reduction="mean")
        bce_EXP = torch.exp(-bce)

        return self.alpha * (1 - bce_EXP) ** self.gamma * bce
