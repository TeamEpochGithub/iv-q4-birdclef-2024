"""Submission sampler the input to the given length."""
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from dask import delayed


@dataclass
class SubmissionSampler:
    """Extract 48(12*4) segments from the input sequence of 4 minutes."""

    @delayed
    def __call__(self, array: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Crop or pad based on length.

        :param array: The input array.
        :return: the cropped or padded array.
        """
        # Raise an error if the audio is not 4 minutes long (32000 samples per second)
        if len(array) != 32000 * 60 * 4:
            raise ValueError("The audio is not 4 minutes long.")
        # Extract 48 segments of 5 seconds each
        step = 32000 * 5
        segments = [array[i : i + step] for i in range(0, len(array), step)]
        return np.array(segments)
