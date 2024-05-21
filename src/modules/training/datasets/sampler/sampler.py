"""Abstract base class for samplers."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt
from dask import delayed


class Sampler(ABC):
    """Abstract base class for samplers."""

    @abstractmethod
    def sample(self, array: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Sample the input array.

        :param array: The input array.
        :return: The sampled array.
        """

    @delayed
    def __call__(self, array: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Apply the sampler as a dask delayed function.

        :param array: The input array.
        :return: The sampled array.
        """
        return self.sample(array)
