"""Post ensembling pipeline with post processing blocks."""


from epochalyst.pipeline.ensemble import EnsemblePipeline
from epochalyst.pipeline.model.training.training import TrainingPipeline

from src.modules.logging.logger import Logger


class PostEnsemble(EnsemblePipeline, TrainingPipeline, Logger):
    """Ensembling with post processing blocks."""
