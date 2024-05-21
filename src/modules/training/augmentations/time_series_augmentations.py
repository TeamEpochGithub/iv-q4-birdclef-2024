"""Time series augmentations for PyTorch tensors."""

from dataclasses import dataclass

import torch


@dataclass
class Scale(torch.nn.Module):
    """Exponential scale 1d augmentation.

    Randomly scales the input signal by a factor sampled from a log-uniform distribution.

    :param p: The probability of applying the augmentation.
    :param lower: The lower bound of the log-uniform distribution.
    :param higher: The higher bound of the log-uniform distribution.
    """

    p: float = 0.5
    lower: float = 1e-5
    higher: float = 1e2

    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the augmentation to the input features and labels.

        :param x: Input features. (N,C,L)|(N,L)
        :param y: Input labels. (N,C)
        :return: The augmented features and labels
        """
        # Randomly sample between 10e-5 and 10e2

        if torch.rand(1) < self.p:
            scale = torch.exp(torch.rand(1) * (torch.log(torch.tensor(self.higher)) - torch.log(torch.tensor(self.lower))) + torch.log(torch.tensor(self.lower)))
            x = x * scale
        return x, y
