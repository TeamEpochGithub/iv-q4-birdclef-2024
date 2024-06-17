"""Merge two consecutive 5s windows into 10s windows."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from typing_extensions import Never

from src.modules.training.verbose_training_block import VerboseTrainingBlock


@dataclass
class Window5sTo10s(VerboseTrainingBlock):
    """Merge two consecutive 5s windows into 10s windows."""

    def custom_train(
        self,
        x: npt.NDArray[np.floating[Any]],
        y: npt.NDArray[np.floating[Any]],
        **train_args: Never,
    ) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
        """Return the input data and labels.

        :param x: The input data in shape (n_samples=48n, n_features=182)
        :param y: The labels in shape (n_samples=48n, n_features=182)
        :param train_args: [UNUSED] The training arguments
        :return: The input data and labels
        """
        return x, y

    def custom_predict(self, x: npt.NDArray[np.floating[Any]], **pred_args: Never) -> npt.NDArray[np.floating[Any]]:
        """Merge two consecutive 5s windows into 10s windows.

        >>> predictions = np.array(
        ...     [
        ...         [0.1, 0.3, 0.3, 0.1],
        ...         [0.2, 0.2, np.nan, 0.9],
        ...         [0.3, 0.1, 0.9, 0.0],
        ...         [0.4, 0.4, 0.0, 0.7],
        ...     ]
        ... )
        >>> Window5sTo10s().custom_predict(predictions)
        array([[0.2, 0.3, 0.3, 0.9],
               [0.2, 0.3, 0.3, 0.9],
               [0.4, 0.4, 0.9, 0.7],
               [0.4, 0.4, 0.9, 0.7]])

        :param x: The input data in shape (n_samples=48n, n_features=182)
        :param pred_args: [UNUSED] The prediction arguments
        :return: The predictions
        """
        windows_to_merge = x.reshape(x.shape[0] // 2, 2, x.shape[1])
        means = np.nanmax(windows_to_merge, axis=1)
        return np.repeat(means, 2, axis=0)
