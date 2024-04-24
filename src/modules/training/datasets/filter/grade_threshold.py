"""Keep samples with grade >= specified score."""


from dataclasses import dataclass
from typing import Any

from src.typing.typing import YData


@dataclass
class GradeThreshold:
    threshold: float = 3.5

    def __call__(self, ydata: YData, year: str) -> Any:
        # Read the metadata for the appropriate year 
        metadata = getattr(ydata, f"meta_{year}")
        # Create a binary mask based on the rating column
        mask = metadata['rating'] >= self.threshold

        # Use the mask to index the labels from ydata
        return getattr(ydata, f"label_{year}").iloc[mask]
