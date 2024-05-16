from dataclasses import dataclass
from typing import Any, Sized, Union

import dask
import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset

from src.modules.training.datasets.dask_dataset import DaskDataset


@dataclass
class DoubleDataset(Dataset):
    """Unsupervised Data Augmentation dataset.

    Wraps around a normal train dataset, but also randomly samples some unlabeled target data every batch."""

    task_dataset: DaskDataset
    discriminator_dataset: DaskDataset

    def __post_init__(self):
        self.short = 'task' if len(self.task_dataset) < len(self.discriminator_dataset) else 'disc'

    def __len__(self):
        """Get the length of the dataset."""
        return min(len(self.task_dataset), len(self.discriminator_dataset))

    def __getitems__(self, indices: list[int]) -> tuple[Any, Any]:
        """Get multiple items from the dataset and apply augmentations if necessary."""
        remapped_indices = np.random.choice(max(len(self.task_dataset), len(self.discriminator_dataset)), len(indices), replace=False)
        if len(self.task_dataset) > len(self):
            task_out = self.task_dataset.__getitems__(remapped_indices)
            disc_out = self.discriminator_dataset.__getitems__(indices)
        else:
            task_out = self.task_dataset.__getitems__(indices)
            disc_out = self.discriminator_dataset.__getitems__(remapped_indices)

        return task_out, disc_out
