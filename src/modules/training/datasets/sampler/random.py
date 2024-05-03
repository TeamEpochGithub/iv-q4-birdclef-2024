"""Select a random fragment of specified length."""
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from dask import delayed


@dataclass
class Random:
    """Select a random fragment of specified length."""

    length: int

    @delayed
    def __call__(self, array: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Select a random fragment of specified length.

        :param array: The input array.
        :return: the randomly selected fragment.
        """
        if len(array) <= self.length:
            # Pad the array
            return np.pad(array, (0, self.length - len(array)))

        start = np.random.randint(0, len(array) - self.length)
        return array[start: start + self.length]