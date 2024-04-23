"""Filter audio segments with too little power."""

from dataclasses import dataclass
import torch

@dataclass
class PSDLabeler:
    threshold = 1

    def __call__(self, x, y):
        # Compute the power
        power = torch.sum(torch.abs(torch.fft.fft(x))**2)
        # Set the silences to 0
        y[power<self.threshold] = 0
        return x, y