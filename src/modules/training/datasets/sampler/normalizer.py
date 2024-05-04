"""Standardize data before applying a sampler."""
from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt
from dask import delayed

from src.modules.training.datasets.sampler.sampler import Sampler


@dataclass
class Normalizer:
    """Standardize data before applying a sampler."""

    sampler: Sampler

    @delayed
    def __call__(self, array: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Crop or pad based on length.

        :param array: The input array.
        :return: the cropped or padded array.
        """
        if array.std() != 0:
            array = array / array.std()
        return self.sampler.sample(array)
