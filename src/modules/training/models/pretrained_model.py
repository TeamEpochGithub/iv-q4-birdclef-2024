"""A module that uses a pretrained model for inference."""
from os import PathLike
from pathlib import Path

import torch


class PretrainedModel(torch.nn.Module):
    """A module that uses a pretrained model for inference.

    This is a somewhat clumsy workaround to use a pretrained model inside a different config.

    :param model: The pretrained model to use.
    :param state_dict_file: The file path to the state dict of the model.
    """
    model: torch.nn.Module
    state_dict_path: Path


    def __init__(self, model: torch.nn.Module, state_dict_path: str | PathLike[str]) -> None:
        """Initialize the model.

        :param model: The pretrained model to use.
        :param state_dict_path: The file path to the state dict of the model.
        """
        super().__init__()
        self.model = model
        self.state_dict_path = Path(state_dict_path)
        self.model.load_state_dict(torch.load(self.state_dict_path))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        :param x: The data of a 4-minute soundscape of shape (48, C, H, W)
        :return: The predictions of shape (48, 182)
        """
        if self.training:
            raise RuntimeError("A pretrained model should not be in training mode during inference.")
        return self.model(x)
