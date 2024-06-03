"""Keep samples with grade >= specified score."""

from dataclasses import dataclass
from typing import Any, cast

import numpy.typing as npt
import pandas as pd

from src.typing.typing import XData, YData


@dataclass
class GradeThreshold:
    """Keep samples with grade >= specified score.

    :param threshold: The threshold to use.
    """

    threshold: float = 3.5

    def __call__(self, xdata: XData, ydata: YData, year: str) -> tuple[npt.NDArray[Any], pd.DataFrame]:
        """Return the filtered/subsampled data.

        :param xdata: The X data.
        :param ydata: The Y data.
        :param year: The "year" to use.
        :return: The filtered data.
        """
        # Read the metadata for the appropriate year
        metadata = ydata[f"meta_{year}"]
        indices = pd.Series()
        # Create a binary mask based on the rating column
        if isinstance(metadata, pd.DataFrame):
            indices = metadata["rating"] >= self.threshold

        # Use the mask to index the labels from ydata
        return cast(npt.NDArray[Any], xdata[f"bird_{year}"])[indices.to_numpy()], ydata[f"label_{year}"].loc[indices]
