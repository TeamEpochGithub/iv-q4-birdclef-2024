"""Submission sampler the input to the given length."""
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from dask import delayed

from src.modules.training.datasets.sampler.sampler import Sampler
from src.utils.logger import logger


@dataclass
class SubmissionSampler(Sampler):
    """Extract 48(12*4) segments from the input sequence of 4 minutes."""

    def sample(self, array: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Extract 48 segments of 5 seconds each from the input sequence of 4 minutes.

        :param array: The input array.
        :return: The array with 48 segments of 5 seconds each.
        """
        # Raise an warning if the audio is not 4 minutes long (32000 samples per second)
        if len(array) != 32000 * 60 * 4:
            logger.warning(f"Audio is not 4 minutes long: {len(array) / 32000} seconds..")
        # Extract 48 segments of 5 seconds each
        step = 32000 * 5
        segments = [array[i : i + step] for i in range(0, len(array), step)]

        # If the length of the last segment is less than 80000 (2.5 sec) remove it, else pad with 0s to 160000
        if len(segments[-1]) < 80000:
            segments = segments[:-1]
        else:
            segments[-1] = np.pad(segments[-1], (0, 160000 - len(segments[-1])), "constant")

        all_segments = np.array(segments)

        # If all segments are not 48, pad with zeros
        if len(all_segments) < 48:
            all_segments = np.pad(all_segments, ((0, 48 - len(all_segments)), (0, 0)), "constant")
        if len(all_segments) > 48:
            all_segments = all_segments[:48]

        # Check that the last segment
        return all_segments

    @delayed
    def __call__(self, array: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Apply the sampler as a dask delayed function."""
        return self.sample(array)
