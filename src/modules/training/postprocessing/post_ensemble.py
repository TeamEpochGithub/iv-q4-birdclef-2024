from dataclasses import dataclass
from typing import Any, Never, TypeVar

import numpy as np
import numpy.typing as npt

from src.modules.training.verbose_training_block import VerboseTrainingBlock

XT = TypeVar('XT', bound=npt.NDArray[np.floating[Any]])
YT = TypeVar('YT', bound=npt.NDArray[np.floating[Any]])


@dataclass
class AlternatingEnsembleModelPredictionsReweight(VerboseTrainingBlock):
    """Reweight the predictions of an alternating ensemble model.

    :param n_models: The number of models used in the ensemble.
    """

    n_models: int


    def custom_train(self,x: XT, y: YT, **train_args: Never) -> tuple[XT, YT]:
        """Do nothing during training. Return the input data and labels.

        :param x: The input data in shape (n_samples=48n, n_features=182)
        :param y: The labels in shape (n_samples=48n, n_features=182)
        :param train_args: (UNUSED) Additional arguments for training
        :return: The input data and labels
        """
        return x, y

    def custom_predict(self, x: XT, **pred_args: Never) -> XT:
        """Reweight the predictions of an alternating ensemble model.

        :param x: The input data in shape (n_samples=48n, n_features=182)
        :param pred_args: (UNUSED) Additional arguments for prediction
        :return: The predictions
        """
        means = np.zeros((self.n_models, x.shape[1]))

        for i in range(self.n_models):
            means[i] = x[:, i::self.n_models].mean(axis=1)

        # TODO(Jeffrey): Implement the reweighting of the predictions

        return x

