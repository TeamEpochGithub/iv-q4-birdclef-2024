"""Timm model for 2D image classification."""
from typing import Any

import torch
from torch import nn

from src.utils.logger import logger


class Timm(nn.Module):
    """Timm model for 2D image classification.

    :param in_channels: Number of input channels
    :param out_channels: Number of output channels
    :param model_name: Model to use
    """

    def __init__(self, in_channels: int, out_channels: int, model_name: str, activation: str | None = None, **kwargs: dict[str, Any]) -> None:
        """Initialize the Timm model.

        :param in_channels: The number of input channels.
        :param out_channels: The number of output channels.
        :param model_name: The model to use.
        :param activation: The activation function to use.
        """
        try:
            import timm  # type: ignore[import-not-found]
        except ImportError as err:
            raise ImportError("Need to install timm if you want to use timm models") from err

        super(Timm, self).__init__()  # noqa: UP008
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation

        # Check if cuda is available
        if torch.cuda.is_available():
            logger.info("CUDA is available")
            try:
                self.model = timm.create_model(model_name, pretrained=True, in_chans=self.in_channels, num_classes=self.out_channels, **kwargs)
            except ValueError:
                logger.warning(f"kwargs {kwargs} not supported for {model_name}. Trying without kwargs.")
                self.model = timm.create_model(model_name, pretrained=True, in_chans=self.in_channels, num_classes=self.out_channels)
        else:
            try:
                self.model = timm.create_model(model_name, pretrained=False, in_chans=self.in_channels, num_classes=self.out_channels, **kwargs)
            except ValueError:
                logger.warning(f"kwargs {kwargs} not supported for {model_name}. Trying without kwargs.")
                self.model = timm.create_model(model_name, pretrained=False, in_chans=self.in_channels, num_classes=self.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Timm model.

        :param x: The input data.
        :return: The output data.
        """
        x = self.model(x)
        # Apply a sigmoid
        if self.activation == "sigmoid":
            return torch.sigmoid(x)
        return x
