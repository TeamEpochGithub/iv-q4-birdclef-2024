"""Computes loss for two outputs from a model"""

import numpy as np
import torch


class FamilyLoss(torch.nn.Module):
    """Custom loss function that computes loss and penalizes bird species predictions."""

    def __init__(self, loss: torch.nn.Module) -> None:
        """Initialize the DoubleLoss class."""
        super().__init__()
        self.loss = loss
        self.family_path = torch.tensor(np.load("data/raw/2024/bird_similarities.npy")).float().to("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the family loss.

        :param inputs: The model predictions.
        :param targets: The true labels.
        :return: The family loss.
        """
        total_loss = self.loss(inputs, targets)

        # From the targets
        target_family = torch.matmul(targets, self.family_path)

        # Weight the predictions
        weighted_loss = total_loss * target_family

        # Compute final loss
        total_loss = torch.sum(weighted_loss)

        return total_loss
