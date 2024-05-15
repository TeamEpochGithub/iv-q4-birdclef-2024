"""Ensemble that alternates between models for inference."""
from collections.abc import Iterable
from typing import Annotated

import torch
from annotated_types import MinLen


class AlternatingEnsemble(torch.nn.Module):
    """Ensemble that alternates between models for inference.

    :param models: The models to alternate between.
    """

    models: Annotated[torch.nn.ModuleList[torch.nn.Module], MinLen(1)]

    def __init__(self, models: Iterable[torch.nn.Module] | None) -> None:
        """Initialize the ensemble.

        :param models: The models to alternate between.
        """
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ensemble.

        :param x: The input data.
        :return: The output data.
        """
        if self.training:
            return torch.stack([model(x) for model in self.models]).mean(dim=0)

        # Alternate inference between model
        out: list[torch.Tensor | None] = [None] * len(x)
        out[0] = self.models[0](x[0])

        for i in range(1, len(x)):
            model = self.models[i % len(self.models)]
            out += model(x[i])

        return torch.stack(out)
