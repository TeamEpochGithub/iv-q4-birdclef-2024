"""Dask dataset module. Torch dataset that works with dask."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import dask
import numpy as np
import numpy.typing as npt
from torch.utils.data import Dataset

from src.typing.typing import XData


@dataclass
class DaskDataset(Dataset):  # type: ignore[type-arg]
    """Main dataset for competition data."""

    X: XData | None = None
    y: npt.NDArray[np.float32] | None = None
    year: str = "2024"
    window: Callable
    use_aug: bool = field(hash=False, repr=False, init=True, default=False)

    def __len__(self) -> int:
        """Get the length of the dataset."""
        # Trick the dataloader into thinking the dataset is smaller than it is
        return len(getattr(self.X, f"bird_{self.year}"))  # type: ignore[arg-type]

    def __getitems__(self, indices: list[int]) -> tuple[Any, Any]:
        """Get multiple items from the dataset and apply augmentations if necessary."""
        x_window = self.window(getattr(self.X, f"bird_{self.year}")[indices])

        x_batch = dask.compute(*x_window)
        y_batch = self.y.iloc[indices]["primary_label"]

        # Apply augmentations if necessary
        if self.augmentations is not None:
            x_tensor, y_tensor = self.augmentations(x_tensor.to("cuda"), y_tensor.to("cuda"))

        return x_tensor, y_tensor
