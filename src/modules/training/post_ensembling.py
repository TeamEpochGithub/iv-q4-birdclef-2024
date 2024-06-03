"""Post ensembling pipeline with post processing blocks."""

from collections.abc import Sequence
from typing import NoReturn

from epochalyst.pipeline.ensemble import EnsemblePipeline
from epochalyst.pipeline.model.model import ModelPipeline
from epochalyst.pipeline.model.training.training import TrainingPipeline
from typing_extensions import Never

from src.modules.logging.logger import Logger


class PostEnsemble(EnsemblePipeline, TrainingPipeline, Logger):
    """Ensembling with post processing blocks.

    :param steps: The models to alternate between.
    """

    steps: Sequence[ModelPipeline]

    def train(self, x: Never, y: Never, **train_args: Never) -> NoReturn:
        """Train the system. This is not implemented for an AlternatingEnsemble.

        :param x: The input to the system.
        :param y: The expected output of the system.
        :param train_args: Keyword arguments.
        :return: The input and output of the system.
        """
        raise NotImplementedError("Cannot train an ensemble. Train the models individually.")
