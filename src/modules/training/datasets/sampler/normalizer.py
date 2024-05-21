"""Standardize data before applying a sampler."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from dask import delayed

from src.modules.training.datasets.sampler.sampler import Sampler


@dataclass
class Normalizer:
    """Standardize data before applying a sampler.

    :param sampler: The sampler to use.
    """

    sampler: Sampler

    @delayed
    def __call__(self, array: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Crop or pad based on length.

        :param array: The input array.
        :return: the cropped or padded array.
        """
        std = array[::1000].std()
        if std != 0:
            array = array / std
        return self.sampler.sample(array)
