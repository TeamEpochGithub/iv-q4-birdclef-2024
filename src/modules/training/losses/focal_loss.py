"""Focal loss implementation, for combating class imbalance in classification tasks."""

from typing import Literal

import torch
import torchvision


class FocalLoss(torch.nn.Module):
    """Focal loss implementation, for combating class imbalance in classification tasks.

    :param alpha: The alpha parameter.
    :param gamma: The gamma parameter.
    :param reduction: The reduction method.
    """

    alpha: float
    gamma: float
    reduction: Literal["none", "mean", "sum"]

    def __init__(self, alpha: float = 0.25, gamma: float = 2,
                 reduction: Literal["none", "mean", "sum"] = "none") -> None:
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


class FocalLossBCEWithLogits(FocalLoss):
    """Focal loss implementation, for combating class imbalance in classification tasks.

    :param bce_weight: The weight of the BCE loss.
    :param focal_weight: The weight of the focal loss.
    """

    bce: torch.nn.BCEWithLogitsLoss

    def __init__(self, alpha: float, ignore_half_labels: bool = False) -> None:
        """Initialize the focal loss.

        :param bce_weight: The weight of the BCE loss.
        :param focal_weight: The weight of the focal loss.
        """
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss(reduction=self.reduction)
        self.alpha = alpha
        self.ignore_half_labels = ignore_half_labels

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the focal loss.

        :param inputs: The model predictions.
        :param targets: The true labels.
        :return: The focal loss.
        """
        bce_loss = self.bce(inputs, targets)
        probas = torch.sigmoid(inputs)

        tmp = targets * self.alpha * (1. - probas) ** self.gamma * bce_loss
        smp = (1. - targets) * probas ** self.gamma * bce_loss

        loss = tmp + smp

        if self.ignore_half_labels:
            loss *= (targets != 0.5)

        loss = loss.mean()
        return loss


class FocalLossBCE(FocalLoss):
    """Focal loss implementation, for combating class imbalance in classification tasks.

    :param bce_weight: The weight of the BCE loss.
    :param focal_weight: The weight of the focal loss.
    """

    bce: torch.nn.BCELoss

    def __init__(self, alpha: float, ignore_half_labels: bool = False) -> None:
        """Initialize the focal loss.

        :param bce_weight: The weight of the BCE loss.
        :param focal_weight: The weight of the focal loss.
        """
        super().__init__()
        self.bce = torch.nn.BCELoss(reduction=self.reduction)
        self.alpha = alpha
        self.ignore_half_labels = ignore_half_labels

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the focal loss.

        :param inputs: The model predictions.
        :param targets: The true labels.
        :return: The focal loss.
        """
        bce_loss = self.bce(inputs, targets)
        probas = inputs

        tmp = targets * self.alpha * (1. - probas) ** self.gamma * bce_loss
        smp = (1. - targets) * probas ** self.gamma * bce_loss
        loss = tmp + smp

        if self.ignore_half_labels:
            loss *= (targets != 0.5)

        loss = loss.mean()
        return loss
