"""Take the first or last segment. Pad if necessary."""
import random
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from src.modules.training.datasets.sampler.sampler import Sampler


@dataclass
class FirstOrLast(Sampler):
    """Take the first or last segment. Pad if necessary."""

    length: int

    def sample(self, array: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Take the first or last segment. Pad if necessary.

        :param array: The input array.
        :return: the cropped or padded array.
        """
        if len(array) < self.length:
            # Pad the array
            return np.pad(array, (0, self.length - len(array)))

        if random.random() > 0.5:
            # Take the first segment
            return array[: self.length]
        # Take the last segment
        return array[-self.length :]
