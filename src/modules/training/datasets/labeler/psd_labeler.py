"""Filter audio segments with too little power."""

from dataclasses import dataclass

import torch


@dataclass
class PSDLabeler:
    """Filter audio segments with too little power.

    :param threshold: The threshold to use.
    """

    threshold = 1

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Set the labels of low power signals to 0.

        :param x: The input tensor.
        :param y: The label tensor.
        :return: The input tensor and the label tensor.
        """
        # Compute the power
        power = torch.sum(torch.abs(torch.fft.fft(x)) ** 2)
        # Set the silences to 0
        y[power < self.threshold] = 0
        return x, y
