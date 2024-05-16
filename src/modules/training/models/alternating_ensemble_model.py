"""Ensemble that alternates between models for inference."""

from collections.abc import Iterable
from typing import Annotated, Final

import torch
from annotated_types import MinLen

N_CLASSES: Final[int] = 182


class AlternatingEnsembleModel(torch.nn.Module):
    """Ensemble that alternates between models for inference.

    Yes, it's an ensemble as a model because our codebase hates any other ensembling techniques than voting.

    :param models: The models to alternate between.
    """

    models: Annotated[torch.nn.ModuleList, MinLen(1)]

    def __init__(self, models: Iterable[torch.nn.Module] | None) -> None:
        """Initialize the ensemble.

        :param models: The models to alternate between.
        """
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ensemble.

        It acts as a fusion ensemble during training, but you probably only want to use this
        for inference with :py:class:`src.modules.training.models.pretrained_model.PretrainedModel`.

        :param x: The data of a 4-minute soundscape of shape (48, C, H, W)
        :return: The predictions of shape (48, 182)
        """
        if self.training:
            return torch.stack([model(x) for model in self.models]).mean(dim=0)

        predictions: torch.Tensor = torch.empty((len(x), N_CLASSES), device=x.device, dtype=x.dtype)

        for i in range(len(self.models)):
            indices = range(i, len(x), len(self.models))
            predictions[indices] = self.models[i](x[indices])

        return predictions
