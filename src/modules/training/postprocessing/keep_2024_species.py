"""Keep only the predictions of the bird species featured in the BirdCLEF 2024 dataset when using the datasets from other years."""
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from typing_extensions import Never

from src.modules.training.verbose_training_block import VerboseTrainingBlock


@dataclass
class Keep2024Species(VerboseTrainingBlock):
    """Keep only the predictions of the bird species featured in the BirdCLEF 2024 dataset when using the datasets from other years."""

    def custom_train(
        self,
        x: npt.NDArray[np.floating[Any]],
        y: npt.NDArray[np.floating[Any]],
        **train_args: Never,
    ) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
        """Return the input data and labels."""
        return x, y

    def custom_predict(self, x: npt.NDArray[np.floating[Any]], **pred_args: Never) -> npt.NDArray[np.floating[Any]]:
        """Keep only the predictions of the bird species featured in the BirdCLEF 2024 dataset.

        Keep in mind that 2024 must be the first year in the input data.

        :param x: The input data in shape (n_samples=48n, n_features=182)
        :return: The predictions
        """
        return x[:, :182]
