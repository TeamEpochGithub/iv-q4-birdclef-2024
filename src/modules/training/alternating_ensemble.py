"""Ensemble that alternates which model to use for prediction."""

from collections.abc import Iterable
from typing import Any, Final, TypeVar, cast

import numpy as np
import numpy.typing as npt

from src.modules.training.post_ensembling import PostEnsemble
from src.typing.typing import XData

SOUND_SIZE: Final[int] = 7680000
WINDOW_SIZE: Final[int] = 160000
NUM_WINDOWS: Final[int] = SOUND_SIZE // WINDOW_SIZE  # = 48

T = TypeVar("T")


def merge_alternating(lists: Iterable[Iterable[T]]) -> list[T]:
    """Merge the lists in an alternating fashion.

    >>> merge_alternating([[1, 2], [3, 4], [5, 6]])
    [1, 3, 5, 2, 4, 6]

    :param lists: The lists to merge.
    :return: The merged list.
    """
    return [x for sublist in zip(*lists, strict=False) for x in sublist]


class AlternatingEnsemble(PostEnsemble):
    """Ensemble that alternates which model to use for prediction."""

    def predict(self, x: XData, **pred_args: Any) -> npt.NDArray[np.floating[Any]]:
        """Predict the input data.

        :param x: The input data.
        :param pred_args: Keyword arguments.
        :return: The predicted data.
        """
        model_data: list[XData] = [XData()] * len(self.steps)

        for year in x.years:
            for i_model in range(len(self.steps)):
                model_windows = [
                    sound.reshape((NUM_WINDOWS, WINDOW_SIZE))[i_window]
                    for sound in cast(npt.NDArray[Any], x[f"bird_{year}"])
                    for i_window in range(i_model, NUM_WINDOWS, len(self.steps))
                ]
                model_data[i_model][f"bird_{year}"] = np.concatenate(model_windows)  # TODO(Jeffrey): Delay the concatenation

        predictions: list[npt.NDArray[np.floating[Any]]] = [model.predict(model_data[i], **pred_args) for i, model in enumerate(self.steps)]
        merged_predictions: list[npt.NDArray[np.floating[Any]]] = [
            np.concatenate(merge_alternating([predictions[i_model].reshape((len(x), NUM_WINDOWS, 182))[i] for i_model in range(len(self.steps))])) for i in range(len(x))
        ]

        return np.concatenate(merged_predictions)
