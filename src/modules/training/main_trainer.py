"""Module for example training block."""
from dataclasses import dataclass
from typing import Any

import pandas as pd
import wandb
from epochalyst.pipeline.model.training.torch_trainer import TorchTrainer

from src.modules.logging.logger import Logger
from src.modules.training.datasets.dask_dataset import DaskDataset
import numpy as np

from src.typing.typing import XData
from src.modules.training.datasets.crop_or_pad import CropOrPad

@dataclass
class MainTrainer(TorchTrainer, Logger):
    """Main training block."""
    year: str = '2024'
    dataset_args: dict[str, Any]

    def save_model_to_external(self) -> None:
        """Save the model to external storage."""
        if wandb.run:
            model_artifact = wandb.Artifact(self.model_name, type="model")
            model_artifact.add_file(f"{self._model_directory}/{self.get_hash()}.pt")
            wandb.log_artifact(model_artifact)

    def create_datasets(self, x: XData, y:pd.DataFrame, train_indices: list[int], test_indices: list[int]):
        train_data = x[train_indices]
        train_labels = y.iloc[train_indices]
        
        test_data = x[test_indices]
        test_labels = y.iloc[test_indices]

        train_dataset = DaskDataset(train_data, train_labels, year=self.year, **self.dataset_args)
        if test_indices is not None:
            test_dataset_args = self.dataset_args.copy()
            del test_dataset_args["augmentations"]
            test_dataset = DaskDataset(test_data, test_labels, year=self.year, **test_dataset_args)
        else:
            test_dataset = None

        return train_dataset, test_dataset
    
