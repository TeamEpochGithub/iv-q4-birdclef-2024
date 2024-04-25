"""Crop or pad the input to the given length."""
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class CropOrPad:
    """Crop or pad the input sequence."""

    length: int

    # @delayed
    def __call__(self, array: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Crop or pad based on length.

        :param array: The input array.
        :return: the cropped or padded array.
        """
        if len(array) < self.length:
            # Pad the array
            return np.pad(array, (0, self.length - len(array)))
        # If len is desired value nothing will happen, else will be cropped
        return array[: self.length]
