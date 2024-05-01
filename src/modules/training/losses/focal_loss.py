"""Focal loss implementation, for combating class imbalance in classification tasks."""

import torch
import torchvision
from torch import nn


class FocalLoss(nn.Module):
    """Focal loss implementation, for combating class imbalance in classification tasks."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2, reduction: str = "mean") -> None:
        """Initialize the focal loss.

        :param alpha: The alpha parameter.
        :param gamma: The gamma parameter.
        :param reduction: The reduction method.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the focal loss.

        :param inputs: The model predictions.
        :param targets: The true labels.

        :return: The focal loss.
        """
        # Flatten label and prediction tensors
        return torchvision.ops.focal_loss.sigmoid_focal_loss(
            inputs=inputs,
            targets=targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )


class FocalLossBCE(FocalLoss):
    """Focal loss implementation, for combating class imbalance in classification tasks."""

    def __init__(self, bce_weight: float = 1.0, focal_weight: float = 1.0) -> None:
        """Initialize the focal loss.

        :param bce_weight: The weight of the BCE loss.
        :param focal_weight: The weight of the focal loss.
        """
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss(reduction=self.reduction)
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the focal loss.

        :param inputs: The model predictions.
        :param targets: The true labels.

        :return: The focal loss.
        """
        # Flatten label and prediction tensors
        # Call forward of the parent
        focall_loss = super().forward(inputs, targets)
        bce_loss = self.bce(inputs, targets)
        return self.bce_weight * bce_loss + self.focal_weight * focall_loss
