"""Keep samples with grade >= specified score."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.typing.typing import XData, YData


@dataclass
class GradeThreshold:
    """Keep samples with grade >= specified score."""

    threshold: float = 3.5

    def __call__(self, xdata: XData, ydata: YData, year: str) -> tuple[XData | npt.NDArray[np.float32], pd.DataFrame]:
        """Return the filtered/subsampled data."""
        # Read the metadata for the appropriate year
        metadata = ydata[f"meta_{year}"]
        indices = pd.DataFrame()
        # Create a binary mask based on the rating column
        if isinstance(metadata, pd.DataFrame):
            indices = metadata["rating"] >= self.threshold

        # Use the mask to index the labels from ydata
        return xdata[f"bird_{year}"][indices.to_numpy()], ydata[f"label_{year}"].loc[indices]  # type: ignore[union-attr]
