"""Abstract base class for samplers."""

from abc import ABC, abstractmethod
from typing import Any

import numpy.typing as npt


class Sampler(ABC):
    """Abstract base class for samplers."""

    @abstractmethod
    def sample(self, array: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Sample the input array.

        :param array: The input array.
        :return: The sampled array.
        """
        pass

