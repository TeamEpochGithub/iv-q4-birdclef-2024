"""Timm model for 2D image classification."""
import torch
from epochalyst.pipeline.model.training.models.timm import Timm as EpochalystTimm


class Timm(EpochalystTimm):
    """Timm model for 2D image classification.

    :param in_channels: Number of input channels
    :param out_channels: Number of output channels
    :param model_name: Model to use
    """

    def __init__(self, in_channels: int, out_channels: int, model_name: str) -> None:
        """Initialize the Timm model.

        :param in_channels: The number of input channels.
        :param out_channels: The number of output channels.
        :param model_name: The model to use.
        """
        super().__init__(in_channels, out_channels, model_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Timm model.

        :param x: The input data.
        :return: The output data.
        """
        x = super().forward(x)
        # Given my chape of (n,c), make sure each column is between 0 and 1
        x = torch.sigmoid(x)
        return x
