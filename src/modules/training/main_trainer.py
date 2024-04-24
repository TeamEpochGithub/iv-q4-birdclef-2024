"""Module for example training block."""
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import wandb
from epochalyst.pipeline.model.training.torch_trainer import TorchTrainer

from src.modules.logging.logger import Logger
from src.modules.training.datasets.dask_dataset import DaskDataset
import numpy as np

from src.typing.typing import XData, YData

@dataclass
class MainTrainer(TorchTrainer, Logger):
    """Main training block."""
    dataset_args: dict[str, Any] = field(default_factory=dict)
    year: str = '2024'
    # Things like prefetch factor
    dataloader_args: dict[str|Any] = field(default_factory=dict)
    

    def save_model_to_external(self) -> None:
        """Save the model to external storage."""
        if wandb.run:
            model_artifact = wandb.Artifact(self.model_name, type="model")
            model_artifact.add_file(f"{self._model_directory}/{self.get_hash()}.pt")
            wandb.log_artifact(model_artifact)

    def create_datasets(self, x: XData, y:YData, train_indices: list[int], test_indices: list[int]):
        # TODO rename to xtrain and ytrain etc.
        train_data = x[train_indices]
        train_labels = y.label_2024.iloc[train_indices]
        
        test_data = x[test_indices]
        test_labels = y.label_2024.iloc[test_indices]

        train_dataset = DaskDataset(X=train_data, y=train_labels, year=self.year, **self.dataset_args)
        if test_indices is not None:
            test_dataset_args = self.dataset_args.copy()
            # TODO fix this
            # test_dataset_args["aug_1d"] = None
            # test_dataset_args["aug_2d"] = None
            test_dataset = DaskDataset(X=test_data, y=test_labels, year=self.year, **test_dataset_args)
        else:
            test_dataset = None

        return train_dataset, test_dataset
    

    def create_dataloaders(
        self,
        train_dataset: Dataset[tuple[Tensor, ...]],
        test_dataset: Dataset[tuple[Tensor, ...]],
    ) -> tuple[DataLoader[tuple[Tensor, ...]], DataLoader[tuple[Tensor, ...]]]:
        """Create the dataloaders for training and validation.

        :param train_dataset: The training dataset.
        :param test_dataset: The validation dataset.
        :return: The training and validation dataloaders.
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=(collate_fn if hasattr(train_dataset, "__getitems__") else None), # type: ignore[arg-type]
            **self.dataloader_args,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=(collate_fn if hasattr(test_dataset, "__getitems__") else None), # type: ignore[arg-type]
            **self.dataloader_args,
        )
        return train_loader, test_loader