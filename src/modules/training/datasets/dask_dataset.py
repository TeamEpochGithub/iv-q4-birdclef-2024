"""Dask dataset module. Torch dataset that works with dask."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import dask
import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset

from src.typing.typing import XData


@dataclass
class DaskDataset(Dataset):  # type: ignore[type-arg]
    """Dask dataset to convert the data to spectrograms."""

    X: XData | None = None
    y: YData | None = None
    year: str = "2024"
    sampler: Callable
    to_2d: Callable | None = None
    filter_: Callable | None = None

    def __post_init__(self):
        # If there is a filter
        # If using torch functions move their parameters to cuda
        if isinstance(self.sampler, torch.nn.Module):
            self.sampler = self.sampler.to('cuda')

    def __len__(self) -> int:
        """Get the length of the dataset."""
        # Trick the dataloader into thinking the dataset is smaller than it is
        return len(getattr(self.y, f"label_{self.year}"))

    def __getitems__(self, indices: list[int]) -> tuple[Any, Any]:
        """Get multiple items from the dataset and apply augmentations if necessary."""
        # TODO Use label .index thing
        x_window = self.sampler(getattr(self.X, f"bird_{self.year}")[indices])

        x_batch = dask.compute(*x_window)
        x_batch = np.stack(x_batch, axis=0)
        x_tensor = torch.from_numpy(x_batch)
        # TODO convert to numpy and fix this
        
        y_batch = getattr(self.y, f"labels_2024").iloc[indices]
        y_batch = y_batch.to_numpy()
        y_tensor = torch.from_numpy(y_batch)

        # Apply augmentations if necessary
        if self.augmentations_1d is not None:
            x_tensor, y_tensor = self.augmentations_1d(x_tensor.to("cuda"), y_tensor.to("cuda"))
        # Convert to 2D if method is specified
        if self.to_2d is not None:
            x_tensor = self.to_2d(x_tensor)
            # Only apply 2D augmentations if converted to 2D
            if self.augmentations_2d is not None:
                x_tensor, y_tensor = self.augmentations_2d(x_tensor, y_tensor)        


        return x_tensor, y_tensor
