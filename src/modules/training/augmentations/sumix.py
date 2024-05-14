"""Sumix implementation"""
from dataclasses import dataclass
import torch

@dataclass
class Sumix:
    """Implementation of Sumix"""
    p: float = 0.5
    max_percent: float = 1.0
    min_percent: float = 0.3

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1) < self.p:
            perm = torch.randperm(x.shape[0])
            coeffs_1 = torch.rand(x.shape[0], device=x.device).view(-1, 1) * (
                self.max_percent - self.min_percent
            ) + self.min_percent
            coeffs_2 = torch.rand(x.shape[0], device=x.device).view(-1, 1) * (
                self.max_percent  - self.min_percent
            ) + self.min_percent

            label_coeffs_1 = torch.where(coeffs_1 >= 0.5, 1, 1 - 2 * (0.5 - coeffs_1))
            label_coeffs_2 = torch.where(coeffs_2 >= 0.5, 1, 1 - 2 * (0.5 - coeffs_2))
            augmented_x = torch.zeros(x.shape)
            augmented_y = torch.zeros(y.shape)
            for i in range(x.shape[0]):
                augmented_x[i] = coeffs_1[i] * x[i] + coeffs_2[i] * x[perm][i]
            for i in range(y.shape[0]):
                augmented_y = torch.clip(label_coeffs_1[i] * y[i] + label_coeffs_2[i] * y[perm][i], 0, 1)
            return augmented_x, augmented_y
        return x, y