"""Reshape the input to the specified output shape."""

import torch
from dataclasses import dataclass

@dataclass
class Reshape:
    """Reshape the input to the specified output shape."""
    shape: list[int]

    def __call__(self, input: torch.Tensor):
        # TODO will implement later for now to test functionality left as no op
        return input
