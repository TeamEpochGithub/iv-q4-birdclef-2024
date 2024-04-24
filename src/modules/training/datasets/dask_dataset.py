"""Dask dataset module. Torch dataset that works with dask."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import dask
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.typing.typing import XData, YData


@dataclass
class DaskDataset(Dataset):  # type: ignore[type-arg]
    """Dask dataset to convert the data to spectrograms."""

    labeler: Callable 
    sampler: Callable
    X: XData | None = None
    # TODO make this work with YData
    y: pd.DataFrame | None = None
    year: str = "2024"
    to_2d: Callable | None = None
    filter_: Callable | None = None
    aug_1d = None
    aug_2d = None   


    def __post_init__(self):
        # ie. keep grade >= 4, 
        if self.filter_ is not None:
            setattr(self.y, f"label_{self.year}", self.filter_(self.y, self.year)) 

        # If using torch functions like Spectrogram, move their parameters to cuda
        if isinstance(self.to_2d, torch.nn.Module):
            self.to_2d = self.to_2d.to('cuda')

        

    def __len__(self) -> int:
        """Get the length of the dataset."""
        # TODO make work with YData
        return len(self.y)

    def __getitems__(self, indices: list[int]) -> tuple[Any, Any]:
        """Get multiple items from the dataset and apply augmentations if necessary."""
        # TODO Use label .index thing
        # Get a window from each sample 
        x_window = []
        for i in indices:
            x_window.append(self.sampler(getattr(self.X, f"bird_{self.year}")[i]))

        x_batch = dask.compute(*x_window)
        x_batch = np.stack(x_batch, axis=0)
        x_tensor = torch.from_numpy(x_batch)
        
        y_batch = self.y.iloc[indices]
        y_batch = y_batch.to_numpy()
        y_tensor = torch.from_numpy(y_batch)

        # Apply augmentations if necessary
        if self.aug_1d is not None:
            x_tensor, y_tensor = self.aug_1d(x_tensor.to("cuda"), y_tensor.to("cuda"))
        # Convert to 2D if method is specified
        if self.to_2d is not None:
            x_tensor = self.to_2d(x_tensor)
            # Only apply 2D augmentations if converted to 2D
            if self.aug_2d is not None:
                x_tensor, y_tensor = self.aug_2d(x_tensor, y_tensor)        


        return x_tensor, y_tensor
