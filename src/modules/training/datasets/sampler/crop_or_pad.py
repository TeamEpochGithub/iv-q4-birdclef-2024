"""Crop or pad the input to the given length."""
from dataclasses import dataclass

import numpy as np
from dask import delayed


@dataclass
class CropOrPad:
    """Crop or pad the input sequence."""

    length: int

    @delayed
    def __call__(self, array: np.ndarray) -> np.ndarray:
        """Crop or pad based on length.

        :param array: The input array.
        :return: the cropped or padded array.
        """
        if len(array) < self.length:
            # Pad the array
            return np.pad(array, (0, self.length - len(array)))
        elif len(array) > self.length:
            # Crop the array
            return array[: self.length]
        else:
            # No need to crop or pad
            return array
