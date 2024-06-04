"""Ensemble with a time limit for predictions."""

import copy
import signal
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt
from agogos.training import TrainType
from epochalyst.pipeline.ensemble import EnsemblePipeline

from src.modules.logging.logger import Logger
from src.utils.timer import ModelTimeoutError, set_alarm

N_CLASSES: Final[int] = 182  # TODO(Jeffrey): Don't hardcode the number of bird species.


@dataclass
class TimedVotingEnsemble(EnsemblePipeline, Logger):
    """Ensemble with a time limit for predictions.

    :param prediction_time: Time limit for predictions.
    """

    prediction_time: int | None = None

    def predict(self, data: npt.NDArray[np.float32], **transform_args: Mapping[Any, Any]) -> npt.NDArray[np.float32]:
        """Transform the input data.

        :param data: The input data of shape (N, C, W, H).
        :return: The predictions of shape (48N, 182).
        """
        out_data: list[npt.NDArray[np.float32]] = [np.full((48 * len(data), N_CLASSES), np.nan)] * len(self.steps)
        if len(self.get_steps()) == 0:
            return data

        if self.prediction_time is not None and self.prediction_time > 0:
            self.log_to_terminal(f"Starting predictions with timeout of {self.prediction_time} seconds.")
            set_alarm(self.prediction_time)

        # Loop through each step and call the transform method
        try:
            for i, step in enumerate(self.get_steps()):
                step_args = transform_args.get(step.__class__.__name__, {})
                out_data[i] = cast(TrainType, step).predict(copy.deepcopy(data), **step_args)
        except TimeoutError as e:
            self.log_to_warning("Ensemble time limit reached. Truncating predictions")
            if isinstance(e, ModelTimeoutError):
                padding = np.subtract((48 * len(data), N_CLASSES), e.predictions.shape)
                out_data[i] = np.pad(e.predictions, ((0, padding[0]), (0, padding[1])), "constant", constant_values=np.nan)
        else:
            signal.alarm(0)

        return np.nanmean(np.array(out_data), axis=0)
