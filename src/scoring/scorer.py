"""Abstract scorer class from which other scorers inherit from."""

from abc import ABC, abstractmethod
from typing import Any

import numpy.typing as npt

from src.typing.typing import YData


class Scorer(ABC):
    """Abstract scorer class from which other scorers inherit from."""

    def __init__(self, name: str) -> None:
        """Initialize the scorer with a name.

        :param name: The name of the scorer.
        """
        self.name = name

    @abstractmethod
    def __call__(self, y_true: YData, y_pred: npt.NDArray[Any], **kwargs: Any) -> dict[str, float]:
        """Calculate the score.

        :param y_true: The true labels.
        :param y_pred: The predicted labels.
        :param kwargs: Additional keyword arguments.
        :return: The calculated score.
        """

    def __str__(self) -> str:
        """Return the name of the scorer.

        :return: The name of the scorer.
        """
        return self.name
