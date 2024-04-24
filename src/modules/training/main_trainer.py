"""Module for example training block."""
from dataclasses import dataclass, field
from typing import Any

import wandb
from epochalyst.pipeline.model.training.torch_trainer import TorchTrainer
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.modules.logging.logger import Logger
from src.modules.training.datasets.dask_dataset import DaskDataset
from src.typing.typing import XData, YData


@dataclass
class MainTrainer(TorchTrainer, Logger):
    """Main training block."""

    dataset_args: dict[str, Any] = field(default_factory=dict)
    year: str = "2024"
    # Things like prefetch factor
    dataloader_args: dict[str, Any] = field(default_factory=dict)

    def save_model_to_external(self) -> None:
        """Save the model to external storage."""
        if wandb.run:
            model_artifact = wandb.Artifact(self.model_name, type="model")
            model_artifact.add_file(f"{self._model_directory}/{self.get_hash()}.pt")
            wandb.log_artifact(model_artifact)

    def create_datasets(self, x: XData, y: YData, train_indices: list[int], test_indices: list[int]) -> tuple[Dataset[tuple[Tensor, ...]], Dataset[tuple[Tensor, ...]]]:
        """Create the datasets for training and validation.

        :param x: The input data.
        :param y: The target variable.
        :param train_indices: The indices to train on.
        :param test_indices: The indices to test on.
        :return: The training and validation datasets.
        """
        x_train = x[train_indices]
        y_train = y[train_indices]

        x_test = x[test_indices]
        y_test = y[test_indices]

        train_dataset = DaskDataset(X=x_train, y=y_train, year=self.year, **self.dataset_args)
        if test_indices is not None:
            test_dataset_args = self.dataset_args.copy()
            # TODO(Tolga): fix this
            # test_dataset_args["aug_1d"] = None
            # test_dataset_args["aug_2d"] = None
            test_dataset = DaskDataset(X=x_test, y=y_test, year=self.year, **test_dataset_args)
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
            collate_fn=(collate_fn if hasattr(train_dataset, "__getitems__") else None),  # type: ignore[arg-type]
            **self.dataloader_args,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=(collate_fn if hasattr(test_dataset, "__getitems__") else None),  # type: ignore[arg-type]
            **self.dataloader_args,
        )
        return train_loader, test_loader


def collate_fn(batch: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    """Collate function for the dataloader.

    :param batch: The batch to collate.
    :return: Collated batch.
    """
    X, y = batch
    return X, y
