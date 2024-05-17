"""Reshape the input to the specified output shape."""

from collections.abc import Iterable
from dataclasses import dataclass

import torch


@dataclass
class Reshape:
    """Reshape the input to the specified output shape.

    :param shape: The desired shape.
    """

    shape: Iterable[int]

    def __call__(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Return a reshaped version of the input tensor."""
        return input_tensor.view(input_tensor.shape[0], 1, *self.shape)
