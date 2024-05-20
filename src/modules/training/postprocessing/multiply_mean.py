"""Module for multiplying the predictions with the mean of the current 4 minute audio file."""
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from src.modules.training.verbose_training_block import VerboseTrainingBlock


@dataclass
class MultiplyMean(VerboseTrainingBlock):
    """Smooth the predictions based on the current 4 minute audio file."""

    power: float = 1

    def custom_train(self, x: npt.NDArray[np.float32], y: npt.NDArray[np.float32], **train_args: Any) -> tuple[Any, Any]:
        """Return the input data and labels."""
        return x, y

    def custom_predict(self, x: npt.NDArray[np.float32], **pred_args: Any) -> npt.NDArray[np.float32]:
        """Apply multiplying the mean to the predictions.

        :param x: The input data in shape (n_samples=48n, n_features=182)
        :return: The predictions
        """
        # Smooth the predictions based on the current 4 minute audio file (48 samples)

        # Calculate the average of the predictions. y_avg will be of size (n, 182)
        x_avg = np.zeros((x.shape[0] // 48, x.shape[1]))
        for i in range(0, x.shape[0], 48):
            sliced = x[i : i + 48]
            x_avg[i // 48] = (sliced**self.power).mean(axis=0) ** (1 / self.power)

        # Loop over all samples and apply the mean
        for i in tqdm(range(x.shape[0]), desc="Smoothing predictions"):
            x[i] = x[i] * x_avg[i // 48]

        return x


# if __name__ == "__main__":
#     # Write a test for smooth kernel
#
#     smooth = SmoothFile(smooth_factor=0.5, kernel=[0.1, 0.2, 0.3, 0.2, 0.1])
#
#     x = np.random.rand(48 * 1, 182)
#     x_smoothed = smooth.smooth_kernel(x)
#     assert x_smoothed.shape == x.shape
