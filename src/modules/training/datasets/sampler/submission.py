"""Submission sampler the input to the given length."""
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from dask import delayed

from src.utils.logger import logger


@dataclass
class SubmissionSampler:
    """Extract 48(12*4) segments from the input sequence of 4 minutes."""

    @delayed
    def __call__(self, array: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Crop or pad based on length.

        :param array: The input array.
        :return: the cropped or padded array.
        """
        # Raise an warning if the audio is not 4 minutes long (32000 samples per second)
        if len(array) != 32000 * 60 * 4:
            logger.warning(f"Audio is not 4 minutes long: {len(array)} samples")
        # Extract 48 segments of 5 seconds each
        step = 32000 * 5
        segments = [array[i : i + step] for i in range(0, len(array), step)]

        # If the length of the last segment is less than 80000 (2.5 sec) remove it, else pad with 0s to 160000
        if len(segments[-1]) < 80000:
            segments = segments[:-1]
        else:
            segments[-1] = np.pad(segments[-1], (0, 160000 - len(segments[-1])), "constant")

        # Check that the last segment
        return np.array(segments)


# Remove delayed for testing
# if __name__ == "__main__":
#     sampler = SubmissionSampler()
#     array = np.random.rand(32000 * 60 * 4)
#     #Add last element with size 32199
#     array = np.append(array, np.random.rand(150001))
#     segments = sampler(array)
#     print(segments)
#     print(len(segments))
