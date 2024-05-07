"""Module for smoothing the predictions based on the current 4 minute audio file."""
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from src.modules.training.verbose_training_block import VerboseTrainingBlock


@dataclass
class SmoothFile(VerboseTrainingBlock):
    """ Smooth the predictions based on the current 4 minute audio file."""

    smooth_factor: float = 0.5


    def custom_train(self, x: npt.NDArray[np.float32], y: npt.NDArray[np.float32], **train_args: Any) -> tuple[Any, Any]:
        """Apply smoothing to the predictions.

        :param x: The input data
        :param y: The target data
        :return: The predictions and the target data
        """




