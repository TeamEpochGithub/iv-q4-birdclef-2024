"""Postprocessing blocks for ensemble models."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from typing_extensions import Never

from src.modules.training.verbose_training_block import VerboseTrainingBlock


@dataclass
class AlternatingEnsembleModelPredictionsReweight(VerboseTrainingBlock):
    """Reweight the predictions of an alternating ensemble model.

    :param n_models: The number of models used in the ensemble.
    """

    n_models: int

    def custom_train(
        self,
        x: npt.NDArray[np.floating[Any]],
        y: npt.NDArray[np.floating[Any]],
        **train_args: Never,
    ) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
        """Do nothing during training. Return the input data and labels.

        :param x: The input data in shape (n_samples=48n, n_features=182)
        :param y: The labels in shape (n_samples=48n, n_features=182)
        :param train_args: (UNUSED) Additional arguments for training
        :return: The input data and labels
        """
        return x, y

    def custom_predict(self, x: npt.NDArray[np.floating[Any]], **pred_args: Never) -> npt.NDArray[np.floating[Any]]:
        """Reweight the predictions of an alternating ensemble model.

        >>> predictions = np.array(
        ...     [
        ...         [0.1, 0.9, 0.1, 0.9, 0.1, 0.9],  # Model 0 mean: 0.2
        ...         [0.3, 0.7, 0.3, 0.7, 0.3, 0.7],  # Model 1 mean: 0.8
        ...     ]
        ... )
        >>> AlternatingEnsembleModelPredictionsReweight(2).custom_predict(predictions)
        array([[0.4, 0.9, 0.4, 0.9, 0.4, 0.9],
               [1.2, 0.7, 1.2, 0.7, 1.2, 0.7]])

        :param x: The input data in shape (n_samples=48n, n_features=182)
        :param pred_args: (UNUSED) Additional arguments for prediction
        :return: The predictions
        """
        means = [x[:, i :: self.n_models].mean() for i in range(self.n_models)]
        highest = max(means)

        for i in range(self.n_models):
            x[:, i :: self.n_models] = x[:, i :: self.n_models] * (highest / means[i])

        return x
