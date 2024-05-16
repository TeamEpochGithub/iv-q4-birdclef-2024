import gc
from abc import abstractmethod
from dataclasses import dataclass

import onnxruntime as onnxrt  # type: ignore[import-not-found]
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.modules.training.datasets.dask_dataset import DaskDataset
from src.modules.training.datasets.double_dataset import UDADataset
from src.modules.training.main_trainer import MainTrainer
from src.typing.typing import XData, YData


@dataclass
class DoubleTrainer(MainTrainer):
    """Unsupervised Data Augmentation Trainer superclass."""

    year_unlabeled: str = "kenya"

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
        losses = []
        self.model.train()
        pbar = tqdm(
            dataloader,
            unit="batch",
            desc=f"Epoch {epoch} Train ({self.initialized_optimizer.param_groups[0]['lr']})",
        )
        for batch in pbar:
            loss = self.one_train_batch(batch)

            # Print tqdm
            losses.append(loss)
            pbar.set_postfix(loss=sum(losses) / len(losses))

        # Step the scheduler
        if self.initialized_scheduler is not None:
            self.initialized_scheduler.step(epoch=epoch)

        # Remove the cuda cache
        torch.cuda.empty_cache()
        gc.collect()

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
        loss = self.criterion(task_pred, ytask, call_pred, ycall)
        
        # Backward pass
        self.initialized_optimizer.zero_grad()
        loss.backward()
        self.initialized_optimizer.step()

        return loss.item()
