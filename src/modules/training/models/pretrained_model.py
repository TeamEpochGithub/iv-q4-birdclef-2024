"""A module that uses a pretrained model for inference."""

from os import PathLike
from pathlib import Path

import torch


class PretrainedModel(torch.nn.Module):
    """A module that uses a pretrained model for inference.

    This is a somewhat clumsy workaround to use a pretrained model inside a different config.

    :param model_path: The path to the pretrained model to load.
    """

    model: torch.nn.Module
    crop_head: int | None = None

    def __init__(self, model_path: str | PathLike[str], crop_head: int | None = None) -> None:
        """Initialize the model.

        :param model_path: The path to the pretrained model to load.
        """
        super().__init__()
        self.model = torch.load(Path(model_path), map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        if isinstance(self.model, torch.nn.DataParallel | torch.nn.parallel.DistributedDataParallel):
            self.model = self.model.module

        self.crop_head = crop_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        :param x: The data of a 4-minute soundscape of shape (48, C, H, W)
        :return: The predictions of shape (48, 182)
        """
        pred = self.model(x)
        if self.crop_head is not None:
            pred = pred[:, : self.crop_head]
        return pred
