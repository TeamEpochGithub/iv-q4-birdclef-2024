"""Ensembles that act like models.

Yes, we use ensembles as models now because our codebase hates any other ensembling techniques than voting.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Annotated, Final

import torch
from annotated_types import MinLen
from typing_extensions import override

N_CLASSES: Final[int] = 182  # TODO(Jeffrey): Don't hardcode the number of bird species.


class EnsembleModel(torch.nn.Module, ABC):
    """Ensemble that acts like model.

    :param models: The models to alternate between.
    """

    models: Annotated[torch.nn.ModuleList, MinLen(1)]

    def __init__(self, models: Iterable[torch.nn.Module] | None) -> None:
        """Initialize the ensemble.

        :param models: The models to use.
        """
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ensemble.

        :param x: The data of a 4-minute soundscape of shape (48, C, H, W)
        :return: The predictions of shape (48, 182)
        """
        raise NotImplementedError("EnsembleModel is an abstract class.")


class FusionEnsembleModel(EnsembleModel):
    """Ensemble that averages the output of the models."""

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ensemble.

        :param x: The data of a 4-minute soundscape of shape (48, C, H, W)
        :return: The predictions of shape (48, 182)
        """
        predictions: list[torch.Tensor] = [torch.empty((x.shape[0], N_CLASSES), device=x.device, dtype=x.dtype)] * len(self.models)

        for i, model in enumerate(self.models):
            predictions[i] = model(x)

        return torch.stack(predictions).nanmean(dim=0)


class AlternatingEnsembleModel(FusionEnsembleModel):
    """Ensemble that alternates between models for inference."""

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ensemble.

        It acts as a fusion ensemble during training, but you probably only want to use this
        for inference with :py:class:`src.modules.training.models.pretrained_model.PretrainedModel`.

        :param x: The data of a 4-minute soundscape of shape (48, C, H, W)
        :return: The predictions of shape (48, 182)
        """
        if self.training:
            return super().forward(x)

        predictions: torch.Tensor = torch.empty((x.shape[0], N_CLASSES), device=x.device, dtype=x.dtype)

        for i, model in enumerate(self.models):
            indices = range(i, x.shape[0], len(self.models))
            predictions[indices] = model(x[indices])

        return predictions
