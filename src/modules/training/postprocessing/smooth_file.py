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
    kernel: list[float] | None = None

    def custom_train(self, x: npt.NDArray[np.float32], y: npt.NDArray[np.float32], **train_args: Any) -> tuple[Any, Any]:
        """Return the input data and labels."""
        return x, y

    def custom_predict(self, x: npt.NDArray[np.float32], **pred_args: Any) -> npt.NDArray[np.float32]:
        """Apply smoothing to the predictions.

        :param x: The input data in shape (n_samples=48n, n_features=182)
        :return: The predictions
        """
        if self.kernel is None:
            return self.smooth_all(x)
        return self.smooth_kernel(x)


    def smooth_all(self, x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Apply smoothing to the predictions.

        :param x: The input data in shape (n_samples=48n, n_features=182)
        :return: The predictions
        """
        # Smooth the predictions based on the current 4 minute audio file (48 samples)

        # Calculate the average of the predictions. y_avg will be of size (n, 182)
        x_avg = np.zeros((x.shape[0] // 48, x.shape[1]))
        for i in range(0, x.shape[0], 48):
            x_avg[i // 48] = np.mean(x[i: i + 48], axis=0)

        # Loop over all samples and apply the smoothing factor
        for i in tqdm(range(x.shape[0]), desc="Smoothing predictions"):
            x[i] = x[i] * (1 - self.smooth_factor) + x_avg[i // 48] * self.smooth_factor

        return x

    def smooth_kernel(self, x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Apply smoothing to the predictions using a kernel.

        :param x: The input data in shape (n_samples=48n, n_features=182)
        :return: The predictions
        """

        # Smooth the predictions based on the current 4 minute audio file (48 samples)
        # Use the self.kernel to smooth the predictions

        x_convolved = np.zeros(x.shape)
        for i in range(0, x.shape[0], 48):
            for channel in range(x.shape[1]):
                x_convolved[i: i + 48, channel] = np.convolve(x[i: i + 48, channel], self.kernel, mode="same")

        # Smooth X using the kernel with the smoothing factor
        x = x * (1 - self.smooth_factor) + x_convolved * self.smooth_factor

        return x




if __name__ == "__main__":
    # Write a test for smooth kernel

    smooth = SmoothFile(smooth_factor=0.5, kernel=[0.1, 0.2, 0.3, 0.2, 0.1])

    x = np.random.rand(48 * 1, 182)
    x_smoothed = smooth.smooth_kernel(x)
    assert x_smoothed.shape == x.shape

