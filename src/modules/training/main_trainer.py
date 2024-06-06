"""Module for example training block."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from threading import Timer
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import onnxruntime as onnxrt
import torch
import wandb
from epochalyst.pipeline.model.training.torch_trainer import TorchTrainer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.modules.logging.logger import Logger
from src.modules.training.datasets.dask_dataset import DaskDataset
from src.modules.training.datasets.sampler.submission import SubmissionSampler
from src.modules.training.models.ensemble_model import EnsembleModel
from src.modules.training.models.two_stage_model import TwoStageModel
from src.modules.training.models.pretrained_model import PretrainedModel
from src.typing.typing import XData, YData

N_CLASSES: Final[int] = 182  # TODO(Jeffrey): Don't hardcode the number of bird species.


@dataclass
class MainTrainer(TorchTrainer, Logger):
    """Main training block.

    :param dataset_args: The arguments for the dataset.
    :param year: The year to use for the dataset.
    :param dataloader_args: The arguments for the dataloader.
    :param weights_path: The path to the weights for the sampler.
    """

    dataset_args: dict[str, Any] = field(default_factory=dict)
    year: str = "2024"
    # Things like prefetch factor
    dataloader_args: dict[str, Any] = field(default_factory=dict, repr=False)
    # Weights for the sampler
    weights_path: str | None = None
    prediction_time: int | None = field(default=None, repr=False)

    model_has_timed_out: bool = field(default=False, init=False, repr=False)

    def save_model_to_external(self) -> None:
        """Save the model to external storage."""
        if wandb.run:
            model_artifact = wandb.Artifact(self.model_name, type="model")
            model_artifact.add_file(f"{self._model_directory}/{self.get_hash()}.pt")
            wandb.log_artifact(model_artifact)

    def create_datasets(
        self,
        x: XData,
        y: YData,
        train_indices: Sequence[int],
        test_indices: Sequence[int],
    ) -> tuple[Dataset[tuple[torch.Tensor, ...]], Dataset[tuple[torch.Tensor, ...]] | None]:
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
        if test_indices:
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
    ) -> Dataset[tuple[torch.Tensor, ...]]:
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

        return DaskDataset(X=x, year="2024", **pred_dataset_args)

    def custom_predict(self, x: Any, **pred_args: Any) -> npt.NDArray[np.float32]:  # noqa: ANN401
        """Predict on the test data.

        Copied form Epochalyst just to change the last line...

        :param x: The input to the system.
        :return: The output of the system.
        """
        self.log_to_debug(f"Predicting model: {self.model.__class__.__name__}")

        # Parse pred_args
        curr_batch_size = pred_args.get("batch_size", self.batch_size)

        # Create dataset
        pred_dataset = self.create_prediction_dataset(x)
        pred_dataloader = DataLoader(
            pred_dataset,
            batch_size=curr_batch_size,
            shuffle=False,
            collate_fn=(collate_fn if hasattr(pred_dataset, "__getitems__") else None),  # type: ignore[arg-type]
        )

        timer: Timer | None = None

        # Set a timeout for the predictions
        if self.prediction_time is not None and self.prediction_time > 0:
            timer = self._set_model_timer(self.prediction_time)

        # Predict with a single model
        if self.n_folds < 1 or pred_args.get("use_single_model", False):
            self._load_model()
            return self.predict_on_loader(pred_dataloader)

        predictions = []
        # Predict with multiple models
        for i in range(int(self.n_folds)):
            if self.model_has_timed_out:
                self.log_to_warning(f"Stopping predictions at fold {i - 1}")
                break

            self._fold = i  # set the fold, which updates the hash
            # Try to load the next fold if it exists
            try:
                self._load_model()
            except FileNotFoundError as e:
                if i == 0:
                    raise FileNotFoundError(f"First model of {self.n_folds} folds not found...") from e
                self.log_to_warning(f"Model for fold {self._fold} not found, skipping the rest of the folds...")
                break
            self.log_to_terminal(f"Predicting with model fold {i + 1}/{self.n_folds}")
            predictions.append(self.predict_on_loader(pred_dataloader))

        if timer is not None:
            timer.cancel()

        # Average the predictions
        test_predictions = np.array(predictions)

        return np.nanmean(test_predictions, axis=0)

    def create_dataloaders(
        self,
        train_dataset: Dataset[tuple[torch.Tensor, ...]],
        test_dataset: Dataset[tuple[torch.Tensor, ...]],
    ) -> tuple[DataLoader[tuple[torch.Tensor, ...]], DataLoader[tuple[torch.Tensor, ...]]]:
        """Create the dataloaders for training and validation.

        :param train_dataset: The training dataset.
        :param test_dataset: The validation dataset.
        :return: The training and validation dataloaders.
        """
        # Check if weights_path exist in self.dataloader_args
        if self.weights_path is not None:
            train_loader = self.create_training_sampler(train_dataset)
        else:
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

    def create_training_sampler(self, train_dataset: Dataset[tuple[torch.Tensor, ...]]) -> DataLoader[tuple[torch.Tensor, ...]]:
        """Create the training sampler for training.

        :param train_dataset: The training dataset.
        :return: The training sampler.
        """
        # Extract targets from dataset
        targets = train_dataset.get_y()  # type: ignore[attr-defined]

        # Take the argmax of the targets
        targets = torch.argmax(targets, dim=1)

        # Create the sampler
        class_weights = np.load(self.weights_path)  # type: ignore[arg-type]

        sample_weights = torch.tensor([class_weights[label] for label in targets])

        sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)  # type: ignore[arg-type]

        loader_args = self.dataloader_args.copy()
        loader_args["sampler"] = sampler

        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            collate_fn=(collate_fn if hasattr(train_dataset, "__getitems__") else None),  # type: ignore[arg-type]
            **loader_args,
        )

    def _load_model(self) -> None:
        """Load the model from the model_directory folder.

        :raises FileNotFoundError: If the model is not found.
        """

        if isinstance(self.model, TwoStageModel) or (
            isinstance(self.model, torch.nn.DataParallel | torch.nn.parallel.DistributedDataParallel) and isinstance(self.model.module, TwoStageModel)
        ):
            self.log_to_terminal("Not loading two-stage model. Make sure to load the individual models.")
            return


        if isinstance(self.model, EnsembleModel) or (
            isinstance(self.model, torch.nn.DataParallel | torch.nn.parallel.DistributedDataParallel) and isinstance(self.model.module, EnsembleModel)
        ):
            self.log_to_terminal("Not loading ensemble model. Make sure to load the individual models.")
            return

        if isinstance(self.model, PretrainedModel) or (
            isinstance(self.model, torch.nn.DataParallel | torch.nn.parallel.DistributedDataParallel) and isinstance(self.model.module, PretrainedModel)
        ):
            self.log_to_terminal("Not loading pretrained model in the main trainer. The model should load in the PretrainedModel class")
            return

        # Check if the model exists
        model_path = Path(self._model_directory) / (self.get_hash() + ".pt")
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found in {model_path.as_posix()}",
            )

        # Load model
        self.log_to_terminal(
            f"Loading model from {model_path.as_posix()}",
        )
        # If device is cuda, load the model to the device
        if self.device == "cuda":
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.load(model_path, map_location="cpu")

        # Load the weights from the checkpoint
        if isinstance(checkpoint, torch.nn.DataParallel | torch.nn.parallel.DistributedDataParallel):
            model = checkpoint.module
        else:
            model = checkpoint

        # Set the current model to the loaded model
        if isinstance(self.model, torch.nn.DataParallel | torch.nn.parallel.DistributedDataParallel):
            self.model.module.load_state_dict(model.state_dict())
        else:
            self.model.load_state_dict(model.state_dict())

        self.log_to_terminal(
            f"Model loaded from {model_path.as_posix()}",
        )

    def predict_on_loader(
        self,
        loader: DataLoader[tuple[torch.Tensor, ...]],
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

        # Predict on the loader
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

    def onnx_predict(self, loader: DataLoader[tuple[torch.Tensor, ...]]) -> npt.NDArray[np.float32]:
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
        predictions: list[npt.NDArray[np.float32]] = [np.full((48, N_CLASSES), np.nan)] * len(loader)

        with torch.no_grad(), tqdm(loader, desc="Predicting", unit="batch", disable=False) as tepoch:
            for data in tepoch:
                if self.model_has_timed_out:
                    self.log_to_warning(f"Stopping predictions at batch {tepoch.n - 1}.")
                    break
                X_batch = data[0].to(self.device).float()
                y_pred = onnx_model.run(output_names, {input_names[0]: X_batch.numpy()})[0]
                predictions[tepoch.n] = y_pred
            else:
                self.log_to_terminal("Done predicting!")

        Path(f"{self.get_hash()}.onnx").unlink()
        return np.concatenate(predictions)

    def _set_model_timer(self, seconds: int) -> Timer:
        """Set the timer for the model to run for the given number of seconds.

        :param seconds: The number of seconds to set the timer for.
        """

        def timeout() -> None:
            """Set the model timed out flag."""
            self.model_has_timed_out = True
            self.log_to_warning("Model has timed out. Finishing current batch.")

        timer = Timer(seconds, timeout)
        timer.start()

        self.log_to_terminal(f"Model timer set for {seconds} seconds.")
        return timer


def collate_fn(batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    """Collate function for the dataloader.

    :param batch: The batch to collate.
    :return: Collated batch.
    """
    X, y = batch
    return X, y
