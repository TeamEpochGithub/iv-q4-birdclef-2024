import gc
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy.typing as npt
import numpy as np
from numpy import float32
from numpy._typing import NDArray
import onnxruntime as onnxrt  # type: ignore[import-not-found]
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.modules.training.datasets.dask_dataset import DaskDataset
from src.modules.training.datasets.double_dataset import DoubleDataset
from src.modules.training.datasets.sampler.submission import SubmissionSampler
from src.modules.training.main_trainer import MainTrainer
from src.typing.typing import XData, YData


@dataclass
class DoubleTrainer(MainTrainer):
    """Unsupervised Data Augmentation Trainer superclass."""

    dataset_args: dict[str, Any] = field(default_factory=dict)
    years: list[str] = field(default_factory=list)
    lambda_: float = 0.0

    def create_datasets(self, x: XData, y: YData, train_indices: list[int], test_indices: list[int]) -> tuple[Dataset[tuple[Tensor, ...]], Dataset[tuple[Tensor, ...]]]:
        x_train = x[train_indices]
        y_train = y[train_indices]

        x_test = x[test_indices]
        y_test = y[test_indices]

        target_dataset_train = DaskDataset(X=x_train, y=y_train, year=self.years[0], **self.dataset_args)  # type: ignore[arg-type]
        disc_dataset_train = DaskDataset(X=x_train, y=y_train, year=self.years[1], **self.dataset_args)  # type: ignore[arg-type]
        if test_indices is not None:
            test_dataset_args = self.dataset_args.copy()
            test_dataset_args["aug_1d"] = None
            test_dataset_args["aug_2d"] = None
            target_dataset_test = DaskDataset(X=x_test, y=y_test, year=self.years[0], **self.dataset_args)  # type: ignore[arg-type]
            disc_dataset_test = DaskDataset(X=x_test, y=y_test, year=self.years[1], **self.dataset_args)  # type: ignore[arg-type]
        else:
            target_dataset_test = None
            disc_dataset_test = None

        train_dataset = DoubleDataset(target_dataset_train, disc_dataset_train)
        test_dataset = DoubleDataset(target_dataset_test, disc_dataset_test)

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

        task_dataset = DaskDataset(X=x, year="2024", **pred_dataset_args)
        return DoubleDataset(task_dataset=task_dataset, discriminator_dataset=None)

    def _train_one_epoch(
            self,
            dataloader: DataLoader[tuple[Tensor, ...]],
            epoch: int,
    ) -> float:
        """Train the model for one epoch.

        :param dataloader: Dataloader for the training data.
        :param epoch: Epoch number.
        :return: Average loss for the epoch.
        """
        task_losses = []
        call_losses = []
        self.model.train()
        pbar = tqdm(
            dataloader,
            unit="batch",
            desc=f"Epoch {epoch} Train ({self.initialized_optimizer.param_groups[0]['lr']})",
        )
        for batch in pbar:
            task_loss, call_loss = self.one_train_batch(batch)

            # Print tqdm
            task_losses.append(task_loss.item())
            call_losses.append(call_loss.item())
            pbar.set_postfix(task_loss=sum(task_losses) / len(task_losses),call_loss=sum(call_losses) / len(call_losses))

        # Step the scheduler
        if self.initialized_scheduler is not None:
            self.initialized_scheduler.step(epoch=epoch)

        # Remove the cuda cache
        torch.cuda.empty_cache()
        gc.collect()
        losses = [task_losses[i] + call_losses[i] for i in range(len(task_losses))]
        return sum(losses) / len(losses)

    def _val_one_epoch(
        self,
        dataloader: DataLoader[tuple[Tensor, ...]],
        desc: str,
    ) -> float:
        """Compute validation loss of the model for one epoch.

        :param dataloader: Dataloader for the testing data.
        :param desc: Description for the tqdm progress bar.
        :return: Average loss for the epoch.
        """
        task_losses = []
        call_losses = []
        self.model.eval()
        pbar = tqdm(dataloader, unit="batch")
        with torch.no_grad():
            for batch in pbar:
                (Xtask, ytask), (Xcall, ycall) = batch
                Xtask = Xtask.to(self.device).float()
                ytask = ytask.to(self.device).float()
                Xcall = Xcall.to(self.device).float()
                ycall = ycall.to(self.device).float()

                # Forward pass
                # Only relevant output is the task for this dataset
                task_pred, _ = self.model(Xtask)
                # Only relevant output is the discriminator (call, nocall)
                _, call_pred = self.model(Xcall)

                # Apply the custom loss function that will apply separate losses to the 2 parts
                task_loss, call_loss = self.criterion(task_pred, ytask, call_pred, ycall)
                # Print losses
                task_losses.append(task_loss.item())
                call_losses.append(call_loss.item())
                pbar.set_description(desc=desc)
                pbar.set_postfix(task_loss=sum(task_losses) / len(task_losses),call_loss=sum((call_losses)) / len(call_losses))
        losses = [task_losses[i] + call_losses[i] for i in range(len(task_losses))]
        return sum(losses) / len(losses)


    def one_train_batch(self, batch: tuple[Tensor, ...]) -> float:
        """Train the model for one batch.

        :param batch: The batch of data.
        :return: The loss for the batch.
        """
        (Xtask, ytask), (Xcall, ycall) = batch
        Xtask = Xtask.to(self.device).float()
        ytask = ytask.to(self.device).float()
        Xcall = Xcall.to(self.device).float()
        ycall = ycall.to(self.device).float()

        # Forward pass
        # Only relevant output is the task for this dataset
        task_pred, _ = self.model(Xtask)
        # Only relevant output is the discriminator (call, nocall)
        _, call_pred = self.model(Xcall)

        # Apply the custom loss function that will apply separate losses to the 2 parts
        task_loss, call_loss = self.criterion(task_pred, ytask, call_pred, ycall)
        loss = task_loss + self.lambda_ * call_loss
        # Backward pass
        self.initialized_optimizer.zero_grad()
        loss.backward()
        self.initialized_optimizer.step()

        return task_loss, call_loss


    def predict_on_loader(
        self,
        loader: DataLoader[tuple[Tensor, ...]],
    ) -> npt.NDArray[np.float32]:
        """Predict on the loader.

        :param loader: The loader to predict on.
        :return: The predictions.
        """
        self.model.eval()
        task_predictions = []
        # disc_predictions = []

        # Create a new dataloader from the dataset of the input dataloader with collate_fn
        if self.device.type == "cuda":
            loader = DataLoader(
                loader.dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=(
                    collate_fn if hasattr(loader.dataset, "__getitems__") else None  # type: ignore[arg-type]
                ),
                **self.dataloader_args,
            )
        else:  # ONNX with CPU
            loader = DataLoader(
                loader.dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=(
                    collate_fn if hasattr(loader.dataset, "__getitems__") else None  # type: ignore[arg-type]
                ),
            )

        if self.device.type == "cuda":
            self.log_to_terminal("Predicting on the test data - Normal")
            with torch.no_grad(), tqdm(loader, unit="batch", disable=False) as tepoch:
                for data in tepoch:
                    (Xtask, ytask), (Xcall, ycall) = data
                    if Xtask is not None:
                        Xtask = Xtask.to(self.device).float()
                    if ytask is not None:
                        ytask = ytask.to(self.device).float()
                    if Xcall is not None:
                        Xcall = Xcall.to(self.device).float()
                    if ycall is not None:
                        ycall = ycall.to(self.device).float()

                    task_pred = self.model(Xtask)[0].squeeze(1).cpu().numpy()
                    # disc_pred = self.model(Xcall)[1].squeeze(1).cpu().numpy()
                    # y_pred = self.model(X_batch).squeeze(1).cpu().numpy()
                    task_predictions.extend(task_pred)
                    # disc_predictions.extend(disc_pred)

            self.log_to_terminal("Done predicting")
            return np.array(task_predictions) # np.array(disc_predictions)
        # ONNX with CPU
        return self.onnx_predict(loader)
    
    def onnx_predict(self, loader: DataLoader[tuple[Tensor, ...]]) -> npt.NDArray[np.float32]:
        """Predict on the loader using ONNX.

        :param loader: The loader to predict on.
        :return: The predictions.
        """
        self.log_to_terminal("Predicting on the test data - ONNX")
        # Get 1 item from the dataloader
        input_tensor = next(iter(loader))[0][0].to(self.device).float()
        input_names = ["actual_input"]
        output_names = ["output"]
        torch.onnx.export(self.model, input_tensor, f"{self.get_hash()}.onnx", verbose=False, input_names=input_names, output_names=output_names)
        onnx_model = onnxrt.InferenceSession(f"{self.get_hash()}.onnx")
        predictions = []
        with torch.no_grad(), tqdm(loader, unit="batch", disable=False) as tepoch:
            for data in tepoch:
                (Xtask, ytask), (Xcall, ycall) = data
                if Xtask is not None:
                    Xtask = Xtask.to(self.device).float()
                if ytask is not None:
                    ytask = ytask.to(self.device).float()
                if Xcall is not None:
                    Xcall = Xcall.to(self.device).float()
                if ycall is not None:
                    ycall = ycall.to(self.device).float()

                y_pred = onnx_model.run(output_names, {input_names[0]: Xtask.numpy()})[0]
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