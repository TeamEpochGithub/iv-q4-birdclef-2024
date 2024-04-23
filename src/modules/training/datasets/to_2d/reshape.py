"""Reshape the input to the specified output shape."""

import torch
from dataclasses import dataclass

@dataclass
class Reshape:
    """Reshape the input to the specified output shape."""
    shape: list[int]

    def __call__(self, input: torch.Tensor):
        
        return input.reshape(input.shape[0], 1, *self.shape)
