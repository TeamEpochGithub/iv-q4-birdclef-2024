"""Example transformation block for the transformation pipeline."""
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from dask import delayed
from tqdm import tqdm

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
            # Check if the attribute exists and is not None
            if hasattr(data, attribute) and getattr(data, attribute) is not None:
                curr_data = getattr(data, attribute)
                for i in tqdm(range(len(curr_data)), desc=f"Transforming {attribute} to zero"):
                    curr_data[i] = self.nan_to_zero(curr_data[i])
        return data

    @delayed
    def nan_to_zero(self, data: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Replace NaN values with 0.

        :param data: The data to transform
        :return: The transformed data
        """
        return np.nan_to_num(data, nan=0.0)


# if __name__ == "__main__":
#     @delayed
#     def test(i):
#         return np.array([i, i, np.nan])
#
#     lazy_data = np.array([test(i) for i in range(10)])
#     X_test = XData(bird_2024=lazy_data, meta_2024=None)
#     block = NanToZero(years=["2024"])
#     transformed_data = block.transform(X_test)
#
#     print(dask.compute(*transformed_data.bird_2024))
