"""Duplicate the predictions of a 10s window to two same 5s windows."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from typing_extensions import Never

from src.modules.training.verbose_training_block import VerboseTrainingBlock


@dataclass
class Window10sTo5s(VerboseTrainingBlock):
    """Duplicate the predictions of a 10s window to two same 5s windows."""

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
        """Duplicate the predictions of a 10s window to two same 5s windows.

        >>> predictions = np.array(
        ...     [
        ...         [0.1, 0.3, 0.3, 0.1],
        ...         [0.2, 0.2, 0.5, 0.9],
        ...     ]
        ... )
        >>> Window10sTo5s().custom_predict(predictions)
        array([[0.1, 0.3, 0.3, 0.1],
               [0.1, 0.3, 0.3, 0.1],
               [0.2, 0.2, 0.5, 0.9],
               [0.2, 0.2, 0.5, 0.9]])

        :param x: The input data in shape (n_samples=48n, n_features=182)
        :param pred_args: [UNUSED] The prediction arguments
        :return: The predictions
        """
        return np.repeat(x, 2, axis=0)
