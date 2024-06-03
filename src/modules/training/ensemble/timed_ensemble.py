import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from agogos.transforming import TransformType
from epochalyst.pipeline.ensemble import EnsemblePipeline
from epochalyst.pipeline.model.training.training import TrainingPipeline
from typing_extensions import override

from src.modules.logging.logger import Logger


@dataclass
class TimedEnsemble(EnsemblePipeline, TrainingPipeline, Logger):
    """Ensemble with a time limit for predictions.

    :param prediction_time: Time limit for predictions.
    """

    prediction_time: float | None = None

    @override
    def transform(self, data: Any, **transform_args: Any) -> Any:
        """Transform the input data.

        :param data: The input data.
        :return: The transformed data.
        """
        # Loop through each step and call the transform method
        out_data = None
        if len(self.get_steps()) == 0:
            return data

        for i, step in enumerate(self.get_steps()):
            step_name = step.__class__.__name__

            step_args = transform_args.get(step_name, {})

            if isinstance(step, TransformType):
                step_data = step.transform(copy.deepcopy(data), **step_args)
                out_data = self.concat(out_data, step_data, self.get_weights()[i])
            else:
                raise TypeError(f"{step} is not an instance of TransformType")

        return out_data

    @override
    def concat(self, original_data: npt.NDArray[np.float32], data_to_concat: npt.NDArray[np.float32], weight: float = 1.0) -> npt.NDArray[np.float32]:
        """Concatenate the trained data.

        :param original_data: First input data
        :param data_to_concat: Second input data
        :param weight: Weight of data to concat
        :return: Concatenated data
        """
        if original_data is None:
            if data_to_concat is None:
                return None
            return data_to_concat * weight

        return original_data + data_to_concat * weight

