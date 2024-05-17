"""Example transformation block for the transformation pipeline."""
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from dask import delayed
from tqdm import tqdm

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock
from src.typing.typing import XData


@dataclass
class Rescale(VerboseTransformationBlock):
    """An example transformation block for the transformation pipeline."""

    years: list[str] = field(default_factory=lambda: ["2024"])
    def custom_transform(self, data: XData) -> XData:
        """Apply a custom transformation to the data.

        :param data: The data to transform
        :param kwargs: Any additional arguments
        :return: The transformed data
        """
        for year in self.years:
            attribute = f"bird_{year}"
            # Check if the attribute exists and is not None
            if hasattr(data, attribute) and getattr(data, attribute) is not None:
                curr_data = getattr(data, attribute)
                for i in tqdm(range(len(curr_data)), desc=f"Rescaling {attribute} to range -1 to 1"):
                    curr_data[i] = self.rescale(curr_data[i])
        return data

    @delayed
    def rescale(self, data: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Rescale the audio to -1 to 1.

        :param data: The data to transform
        :return: The transformed data
        """
        return 2 * ((data - data.min()) / (data.max() - data.min())) - 1


    def __call__(self, data: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Replace NaN values with 0.

        :param data: The data to transform
        :return: The transformed data
        """
        return self.rescale(data)

