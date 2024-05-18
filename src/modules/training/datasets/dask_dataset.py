"""Dask dataset module. Torch dataset that works with dask."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, cast

import dask
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.typing.typing import XData, YData


@dataclass
class DaskDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dask dataset to convert the data to spectrograms.

    :param labeler: The function to label the data.
    :param sampler: The function to sample the data.
    :param process_delayed: The delayed transformations to apply to the data.
    :param X: The X data.
    :param y: The y data.
    :param year: The "year" to use.
    :param to_2d: The function to convert the data to 2D.
    :param filter_: The function to filter the data.
    :param aug_1d: The 1D augmentations to apply.
    :param aug_2d: The 2D augmentations to apply.
    """

    labeler: Callable[[torch.Tensor], torch.Tensor]
    sampler: Callable[[npt.NDArray[Any]], npt.NDArray[Any]]

    process_delayed: Iterable[Callable[[npt.NDArray[Any]], npt.NDArray[Any]]] = field(default_factory=list)

    X: XData | None = None
    y: YData | None = None
    year: str = "2024"
    to_2d: Callable[[torch.Tensor], torch.Tensor] | None = None
    filter_: Callable[[XData | None, YData | None, str], tuple[npt.NDArray[Any], pd.DataFrame]] | None = None
    aug_1d: Callable[[torch.Tensor, torch.Tensor | None], tuple[torch.Tensor, torch.Tensor | None]] | None = None
    aug_2d: Callable[[torch.Tensor, torch.Tensor | None], tuple[torch.Tensor, torch.Tensor | None]] | None = None

    def __post_init__(self) -> None:
        """Filter the data if filter_ is specified."""
        # ie. keep grade >= 4,
        if self.filter_ is not None and self.X is not None and self.y is not None:
            filtered_x, filtered_y = self.filter_(self.X, self.y, self.year)
            self.y[f"label_{self.year}"] = filtered_y
            self.X[f"bird_{self.year}"] = filtered_x

    def __len__(self) -> int:
        """Get the length of the dataset.

        :return: The length of the dataset.
        """
        return 0 if self.X is None else len(self.X[f"bird_{self.year}"])

    def get_y(self) -> torch.Tensor:
        """Get the labels.

        :return: The labels.
        """
        return torch.from_numpy(self.y[f"label_{self.year}"].to_numpy()) if self.y is not None else torch.tensor([])

    def __getitems__(self, indices: Iterable[int]) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Get multiple items from the dataset and apply augmentations if necessary.

        :param indices: The indices to get.
        :return: The X and y data.
        """
        # Get a window from each sample

        if self.X is not None:
            x_window = [self.sampler(cast(npt.NDArray[Any], self.X[f"bird_{self.year}"])[i]) for i in indices]

            # Apply any delayed transformations
            for step in self.process_delayed:
                x_window = [step(x) for x in x_window]

        x_batch = dask.compute(*x_window)

        x_batch = np.stack(x_batch, axis=0)

        # If the x_batch is 3D, convert to 2D
        if len(x_batch.shape) == 3:
            x_batch = x_batch.reshape(x_batch.shape[0] * x_batch.shape[1], x_batch.shape[2])

        x_tensor = torch.from_numpy(x_batch)
        y_tensor = None

        if self.y is not None and isinstance(self.y[f"label_{self.year}"], pd.DataFrame):
            y_batch = self.y[f"label_{self.year}"].iloc[indices]  # type: ignore[call-overload]
            y_tensor = torch.from_numpy(y_batch.to_numpy())

        if self.aug_1d is not None:
            x_tensor, y_tensor = self.aug_1d(x_tensor.unsqueeze(1), y_tensor)
        # Convert to 2D if method is specified
        if self.to_2d is not None:
            x_tensor = self.to_2d(x_tensor)
            # Only apply 2D augmentations if converted to 2D
            if self.aug_2d is not None:
                x_tensor, y_tensor = self.aug_2d(x_tensor, y_tensor)

        return x_tensor, y_tensor
