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


    def __len__(self):
        """Get the length of the dataset."""
        return len(self.task_dataset)

    def __getitems__(self, indices: list[int]) -> tuple[Any, Any]:
        """Get multiple items from the dataset and apply augmentations if necessary."""
        task_out = self.task_dataset.__getitems__(indices)
        if self.discriminator_dataset is not None:
            remapped_indices = np.random.choice(len(self.discriminator_dataset), len(indices), replace=False)
            disc_out = self.discriminator_dataset.__getitems__(remapped_indices)
        else:
            return task_out, (None, None)
        return task_out, disc_out
