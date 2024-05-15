"""Select a random fragment of specified length."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from src.modules.training.datasets.sampler.sampler import Sampler


@dataclass
class Random(Sampler):
    """Select a random fragment of specified length."""

    length: int

    def sample(self, array: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Select a random fragment of specified length.

        :param array: The input array.
        :return: the randomly selected fragment.
        """
        if len(array) <= self.length:
            # Pad the array
            return np.pad(array, (0, self.length - len(array)))

        gen = np.random.default_rng()
        start = gen.integers(0, len(array) - self.length)
        return array[start : start + self.length]
