"""Example transformation block for the transformation pipeline."""
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from dask import delayed

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock
from src.typing.typing import XData


@dataclass
class NanToZero(VerboseTransformationBlock):
    """An example transformation block for the transformation pipeline."""

    years: list[str]

    def custom_transform(self, data: XData) -> XData:
        """Apply a custom transformation to the data.

        :param data: The data to transform
        :param kwargs: Any additional arguments
        :return: The transformed data
        """
        for year in self.years:
            attribute = f"bird_{year}"
            if hasattr(data, attribute):
                setattr(data, attribute, self.nan_to_zero(getattr(data, attribute)))
        return data

    @delayed
    def nan_to_zero(self, data: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Replace NaN values with 0.

        :param data: The data to transform
        :return: The transformed data
        """
        return np.nan_to_num(data, nan=0.0)
