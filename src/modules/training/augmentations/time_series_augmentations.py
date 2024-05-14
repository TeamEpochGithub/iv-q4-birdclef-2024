"""Time series augmentations for PyTorch tensors."""

from dataclasses import dataclass

import torch


@dataclass
class Scale(torch.nn.Module):
    """Exponential scale 1d augmentation.

    Randomly scales the input signal by a factor sampled from a log-uniform distribution.
    """

    p: float = 0.5

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
            scale = 10 ** (torch.rand(1) * 7 - 5)
            x = x * scale
        return x, y


# if __name__ == "__main__":
#     # Test the Scale augmentation
#     scale = Scale(p=1)
#     x = torch.rand(10, 1, 100)
#     y = torch.rand(10, 1)
#     x_aug, y_aug = scale(x, y)
#     assert x_aug.shape == x.shape
#     assert y_aug.shape == y.shape
#     assert torch.allclose(x_aug / x, y_aug / y, atol=1e-6)
#     print("Scale augmentation test passed.")
