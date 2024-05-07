"""Module for smoothing the predictions based on the current 4 minute audio file."""
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from src.modules.training.verbose_training_block import VerboseTrainingBlock


@dataclass
class SmoothFile(VerboseTrainingBlock):
    """Smooth the predictions based on the current 4 minute audio file."""

    smooth_factor: float = 0.1

    def custom_train(self, x: npt.NDArray[np.float32], y: npt.NDArray[np.float32], **train_args: Any) -> tuple[Any, Any]:
        """Return the input data and labels."""
        return x, y

    def custom_predict(self, x: npt.NDArray[np.float32], **pred_args: Any) -> npt.NDArray[np.float32]:
        """Apply smoothing to the predictions.

        :param x: The input data in shape (n_samples=48n, n_features=182)
        :return: The predictions
        """
        # Smooth the predictions based on the current 4 minute audio file (48 samples)

        # Calculate the average of the predictions. y_avg will be of size (n, 182)
        x_avg = np.zeros((x.shape[0] // 48, x.shape[1]))
        for i in range(0, x.shape[0], 48):
            x_avg[i // 48] = np.mean(x[i : i + 48], axis=0)

        # Loop over all samples and apply the smoothing factor
        for i in tqdm(range(x.shape[0]), desc="Smoothing predictions"):
            x[i] = x[i] * (1 - self.smooth_factor) + x_avg[i // 48] * self.smooth_factor

        return x
