"""Module for example training block."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import onnxruntime as onnxrt  # type: ignore[import-not-found]
import torch
import wandb
from epochalyst.pipeline.model.training.torch_trainer import TorchTrainer
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.modules.logging.logger import Logger
from src.modules.training.datasets.dask_dataset import DaskDataset
from src.modules.training.datasets.sampler.submission import SubmissionSampler
from src.typing.typing import XData, YData


@dataclass
class MainTrainer(TorchTrainer, Logger):
    """Main training block."""

    dataset_args: dict[str, Any] = field(default_factory=dict)
    year: str = "2024"
    # Things like prefetch factor
    dataloader_args: dict[str, Any] = field(default_factory=dict, repr=False)

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
            test_dataset_args["aug_1d"] = None
            test_dataset_args["aug_2d"] = None
            test_dataset = DaskDataset(X=x_test, y=y_test, year=self.year, **test_dataset_args)
        else:
            test_dataset = None

        return train_dataset, test_dataset

    def create_prediction_dataset(
        self,
        x: XData,
    ) -> Dataset[tuple[Tensor, ...]]:
        """Create the prediction dataset for submission used in custom_predict.

        :param x: The input data.
        :return: The prediction dataset.
        """
        pred_dataset_args = self.dataset_args.copy()
        if pred_dataset_args.get("aug_1d") is not None:
            del pred_dataset_args["aug_1d"]
        if pred_dataset_args.get("aug_2d") is not None:
            del pred_dataset_args["aug_2d"]
        pred_dataset_args["sampler"] = SubmissionSampler()

        return DaskDataset(X=x, year=self.year, **pred_dataset_args)

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

    def _load_model(self) -> None:
        """Load the model from the model_directory folder."""
        # Check if the model exists
        if not Path(f"{self._model_directory}/{self.get_hash()}.pt").exists():
            raise FileNotFoundError(
                f"Model not found in {self._model_directory}/{self.get_hash()}.pt",
            )

        # Load model
        self.log_to_terminal(
            f"Loading model from {self._model_directory}/{self.get_hash()}.pt",
        )
        # If device is cuda, load the model to the device
        if self.device == "cuda":
            checkpoint = torch.load(f"{self._model_directory}/{self.get_hash()}.pt")
        else:
            checkpoint = torch.load(f"{self._model_directory}/{self.get_hash()}.pt", map_location="cpu")

        # Load the weights from the checkpoint
        if isinstance(checkpoint, nn.DataParallel):
            model = checkpoint.module
        else:
            model = checkpoint

        # Set the current model to the loaded model
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(model.state_dict())
        else:
            self.model.load_state_dict(model.state_dict())

        self.log_to_terminal(
            f"Model loaded from {self._model_directory}/{self.get_hash()}.pt",
        )

    def predict_on_loader(
        self,
        loader: DataLoader[tuple[Tensor, ...]],
    ) -> npt.NDArray[np.float32]:
        """Predict on the loader.

        :param loader: The loader to predict on.
        :return: The predictions.
        """
        self.model.eval()
        predictions = []
        # Create a new dataloader from the dataset of the input dataloader with collate_fn

        if self.device.type == "cuda":
            loader = DataLoader(
                loader.dataset,
                batch_size=loader.batch_size,
                shuffle=False,
                collate_fn=(
                    collate_fn if hasattr(loader.dataset, "__getitems__") else None  # type: ignore[arg-type]
                ),
                **self.dataloader_args,
            )
        else:  # ONNX with CPU
            loader = DataLoader(
                loader.dataset,
                batch_size=loader.batch_size,
                shuffle=False,
                collate_fn=(
                    collate_fn if hasattr(loader.dataset, "__getitems__") else None  # type: ignore[arg-type]
                ),
            )

        if self.device.type == "cuda":
            self.log_to_terminal("Predicting on the test data - Normal")
            with torch.no_grad(), tqdm(loader, unit="batch", disable=False) as tepoch:
                for data in tepoch:
                    X_batch = data[0].to(self.device).float()

                    y_pred = self.model(X_batch).squeeze(1).cpu().numpy()
                    predictions.extend(y_pred)

            self.log_to_terminal("Done predicting")
            return np.array(predictions)
        # ONNX with CPU
        return self.onnx_predict(loader)

    def onnx_predict(self, loader: DataLoader[tuple[Tensor, ...]]) -> npt.NDArray[np.float32]:
        """Predict on the loader using ONNX.

        :param loader: The loader to predict on.
        :return: The predictions.
        """
        self.log_to_terminal("Predicting on the test data - ONNX")
        # Get 1 item from the dataloader
        input_tensor = next(iter(loader))[0].to(self.device).float()
        input_names = ["actual_input"]
        output_names = ["output"]
        torch.onnx.export(self.model, input_tensor, f"{self.get_hash()}.onnx", verbose=False, input_names=input_names, output_names=output_names)
        onnx_model = onnxrt.InferenceSession(f"{self.get_hash()}.onnx")
        predictions = []
        with torch.no_grad(), tqdm(loader, unit="batch", disable=False) as tepoch:
            for data in tepoch:
                X_batch = data[0].to(self.device).float()
                y_pred = onnx_model.run(output_names, {input_names[0]: X_batch.numpy()})[0]
                predictions.extend(y_pred)

        self.log_to_terminal("Done predicting")
        # Remove the saved onnx model
        Path(f"{self.get_hash()}.onnx").unlink()
        return np.array(predictions)


def collate_fn(batch: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    """Collate function for the dataloader.

    :param batch: The batch to collate.
    :return: Collated batch.
    """
    X, y = batch
    return X, y
