"""Utility functions for setting prediction timers and returning partial results."""

import signal
from types import FrameType
from typing import Any, NoReturn

import numpy.typing as npt
import torch


def handle_timeout(signum: int, frame: FrameType | None) -> NoReturn:  # noqa: ARG001
    """Handle the timeout signal.

    :raise TimeoutError: If the timeout is reached.
    """
    raise TimeoutError("Time limit reached")


def set_alarm(seconds: int) -> None:
    """Set the alarm for the given number of seconds.

    :param seconds: The number of seconds to set the alarm for.
    """
    signal.signal(signal.SIGALRM, handle_timeout)
    signal.alarm(seconds)


class ModelTimeoutError(TimeoutError):
    """Error for when the model runs out of time, but has a partial prediction.

    :param predictions: The predictions made before the timeout.
    """

    predictions: npt.NDArray[Any] | torch.Tensor

    def __init__(self, message: str, predictions: npt.NDArray[Any] | torch.Tensor) -> None:
        """Initialize the error.

        :param message: The message of the error.
        :param predictions: The predictions made before the timeout.
        """
        super().__init__(message)
        self.predictions = predictions

    def __str__(self) -> str:
        """Return the string representation of the error."""
        return f"{super().__str__()} {len(self.predictions)} predictions were made."
