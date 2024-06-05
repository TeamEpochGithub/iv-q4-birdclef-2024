"""Timm model for 2D image classification."""

from typing import Final

import torch
from typing_extensions import override

N_CLASSES: Final[int] = 182  # TODO(Jeffrey): Don't hardcode the number of bird species.


class TwoStageModel(torch.nn.Module):
    """Two-stage model that predicts both call and nocall.

    :param model1: The first model to use (for prediction call / nocall).
    :param model2: The second model to use (for prediction bird species).
    """

    model1: torch.nn.Module
    model2: torch.nn.Module
    threshold: float

    def __init__(self, model1: torch.nn.Module, model2: torch.nn.Module, threshold: float) -> None:
        """Initialize the ensemble.

        :param model1: The first model to use (for prediction call / nocall).
        :param model2: The second model to use (for prediction bird species).
        :param threshold: The threshold to use for the first model.
        """
        super().__init__()
        self.model1 = model1
        self.model2 = model2
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the 2-stage model.

        :param x: The data of a 4-minute soundscape of shape (48, C, H, W)
        :return: The predictions of shape (48, 182)
        """
        # Start by feeding forward the first model
        preds = self.model1(x)
        preds2 = self.model2(x)

        return preds * preds2


        # Output of preds is now (48, 1)
        # We need to convert this to a binary classification to check if it is a call or not. Use the self.thresholdto determine if it is a call or not for every item in the batch
        calls = preds > self.threshold

        #Get the indices of the calls
        call_indices = calls.nonzero(as_tuple=True)[0]

        #print(len(call_indices) / 24)

        # Get the predictions of the second model
        preds = self.model2(x[call_indices])

        # Create a tensor of zeros with the same shape as the output of the second model
        final_preds = torch.zeros((x.shape[0], N_CLASSES), device=x.device, dtype=x.dtype)

        # Fill in the predictions of the second model
        final_preds[call_indices] = preds

        return final_preds