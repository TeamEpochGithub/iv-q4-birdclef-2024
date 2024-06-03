"""Sumix implementation copied from Kaggle."""

from dataclasses import dataclass

import torch


@dataclass
class Sumix:
    """Implementation of Sumix class.

    :param p: The probability of applying the augmentation
    :param max_percent: The maximum percentage to mix
    :param min_percent: The minimum percentage to mix
    """

    p: float = 0.5
    max_percent: float = 1.0
    min_percent: float = 0.3

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply sumix augmentation to the input. A different type of mixup that uses more aggressive weights for the labels.

        :param x: Input tensor (N,C,L)
        :param y: Input labels
        :return: The augmented input tensor and label tensor
        """
        if torch.rand(1) < self.p:
            perm = torch.randperm(x.shape[0])
            coeffs_1 = torch.rand(x.shape[0], device=x.device).view(-1, 1) * (self.max_percent - self.min_percent) + self.min_percent
            coeffs_2 = torch.rand(x.shape[0], device=x.device).view(-1, 1) * (self.max_percent - self.min_percent) + self.min_percent

            label_coeffs_1 = torch.where(coeffs_1 >= 0.5, 1, 1 - 2 * (0.5 - coeffs_1))
            label_coeffs_2 = torch.where(coeffs_2 >= 0.5, 1, 1 - 2 * (0.5 - coeffs_2))

            augmented_x = coeffs_1.unsqueeze(-1).expand(x.shape) * x + coeffs_2.unsqueeze(-1).expand(x.shape) * x[perm]
            augmented_y = torch.clip(label_coeffs_1.expand(y.shape) * y + label_coeffs_2.expand(y.shape) * y[perm], 0, 1)
            return augmented_x, augmented_y
        return x, y
