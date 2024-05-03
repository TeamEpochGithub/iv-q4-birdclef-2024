"""Weighted BCE loss implementation, for combating class imbalance in classification tasks."""
import numpy as np
import torch
from torch import nn

from src.utils.logger import logger


class WeightedLoss(nn.Module):
    """Weighted BCE Loss implementation, for combating class imbalance in classification tasks."""

    def __init__(self, weights_path: str) -> None:
        """Initialize the weighted BCE loss.

        :param weights_path: The path to the weights file.

        """
        super().__init__()
        # Push weights to cuda
        try:
            self.weights = torch.from_numpy(np.load(weights_path)).cuda()
        except FileNotFoundError:
            logger.warning(f"Could not find the weights file at {weights_path}. Using default weights.")

    # Have an abstract forward
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the weighted BCE Loss.

        :param inputs: The model predictions.
        :param targets: The true labels.

        :return: The focal loss.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class WeightedBCEWithLogitsLoss(WeightedLoss):
    """Weighted BCE loss implementation with logits, for combating class imbalance in classification tasks."""

    def __init__(self, weights_path: str) -> None:
        """Initialize the weighted BCE loss.

        :param weights_path: The path to the weights file.

        """
        super().__init__(weights_path)

        self.bce = torch.nn.BCEWithLogitsLoss(weight=self.weights)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the weighted BCE Loss.

        :param inputs: The model predictions.
        :param targets: The true labels.

        :return: The focal loss.
        """
        # Flatten label and prediction tensors
        return self.bce(inputs, targets)


class WeightedBCELoss(WeightedLoss):
    """Weighted BCE loss implementation, for combating class imbalance in classification tasks."""

    def __init__(self, weights_path: str) -> None:
        """Initialize the weighted BCE loss.

        :param weights_path: The path to the weights file.

        """
        super().__init__(weights_path)
        self.bce = torch.nn.BCELoss(weight=self.weights)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the weighted BCE Loss.

        :param inputs: The model predictions.
        :param targets: The true labels.

        :return: The focal loss.
        """
        # Flatten label and prediction tensors
        return self.bce(inputs, targets)
