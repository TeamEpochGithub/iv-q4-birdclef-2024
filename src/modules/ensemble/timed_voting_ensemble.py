"""Ensemble with a time limit for predictions."""

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from threading import Timer
from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt
from agogos.training import TrainType
from epochalyst.pipeline.ensemble import EnsemblePipeline
from epochalyst.pipeline.model.model import ModelPipeline

from src.modules.logging.logger import Logger

N_CLASSES: Final[int] = 182  # TODO(Jeffrey): Don't hardcode the number of bird species.


@dataclass
class TimedVotingEnsemble(EnsemblePipeline, Logger):
    """Ensemble with a time limit for predictions.

    :param prediction_time: Time limit for predictions.
    """

    prediction_time: int | None = None

    ensemble_has_timed_out: bool = field(default=False, init=False, repr=False)
    cur_step: int = field(default=0, init=False, repr=False)
    post_process: list[Any] = None

    def predict(self, data: npt.NDArray[np.float32], **transform_args: Mapping[Any, Any]) -> npt.NDArray[np.float32]:
        """Transform the input data.

        :param data: The input data of shape (N, C, W, H).
        :return: The predictions of shape (48N, 182).
        """
        out_data: list[npt.NDArray[np.float32]] = [np.full((48 * len(data), N_CLASSES), np.nan)] * len(self.steps)
        if len(self.get_steps()) == 0:
            return data

        timer: Timer | None = None

        if self.prediction_time is not None and self.prediction_time > 0:
            timer = self._set_ensemble_timer(self.prediction_time)

        # Loop through each step and call the transform method
        for i, step in enumerate(self.get_steps()):
            self.cur_step = i
            if self.ensemble_has_timed_out:
                self.log_to_warning(f"Stopping predictions at model {i - 1}")
                timer.cancel()
                break
            step_args = transform_args.get(step.__class__.__name__, {})
            out_data[i] = cast(TrainType, step).predict(copy.deepcopy(data), **step_args) * self.weights[i]

        if timer:
            timer.cancel()

        curr_mean = (np.nanmean(np.array(out_data), axis=0) / np.sum(self.weights)) * len(self.weights)
        if self.post_process:
            for el in self.post_process:
                pred_args = transform_args.get("ModelPipeline", {}).get("train_sys", {}).get(el.__class__.__name__, {})
                curr_mean = el.custom_predict(curr_mean, **pred_args)

        return curr_mean

    def _set_ensemble_timer(self, seconds: int) -> Timer:
        """Set the timer for the ensemble to run for the given number of seconds.

        :param seconds: The number of seconds to set the timer for.
        """

        def timeout() -> None:
            """Set the ensemble timed out flag."""
            self.ensemble_has_timed_out = True

            # Attempt to set the model_has_timed_out flag for the current model.
            cur_model_pipeline = cast(ModelPipeline, self.get_steps()[self.cur_step])
            cur_trainer = cur_model_pipeline.train_sys.steps[0]  # TODO(Jeffrey): Find a cleaner way to access the Trainer
            cur_trainer.model_has_timed_out = True

            self.log_to_warning("Ensemble has timed out. Finishing current model.")

        timer = Timer(seconds, timeout)
        timer.start()

        self.log_to_terminal(f"Ensemble timer set for {seconds} seconds.")
        return timer
