"""Dummy models for testing purposes."""

from typing import Final

import torch

N_CLASSES: Final[int] = 182  # TODO(Jeffrey): Don't hardcode the number of bird species.


class NanModel(torch.nn.Module):
    """A model that always predicts NaN."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        :param x: The data of a 4-minute soundscape of shape (48, C, H, W)
        :return: The predictions of shape (48, 182)
        """
        return torch.empty((x.shape[0], N_CLASSES), device=x.device, dtype=x.dtype)


class ZeroModel(torch.nn.Module):
    """A model that always predicts 0."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        :param x: The data of a 4-minute soundscape of shape (48, C, H, W)
        :return: The predictions of shape (48, 182)
        """
        return torch.zeros((x.shape[0], N_CLASSES), device=x.device, dtype=x.dtype)


class RandomModel(torch.nn.Module):
    """A model that always predicts random values."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        :param x: The data of a 4-minute soundscape of shape (48, C, H, W)
        :return: The predictions of shape (48, 182)
        """
        return torch.rand((x.shape[0], N_CLASSES), device=x.device, dtype=x.dtype)
