"""Common type definitions for the project."""
from dataclasses import dataclass
from typing import Any

import numpy.typing as npt
import pandas as pd


@dataclass
class XData:
    """Dataclass to hold X data.

    :param meta_2024: Metadata of BirdClef2024:
    :param meta_2023: Metadata of BirdClef2023:
    :param meta_2022: Metadata of BirdClef2022:
    :param meta_2021: Metadata of BirdClef2021:
    :param bird_2024: Audiodata of BirdClef2024
    :param bird_2023: Audiodata of BirdClef2023
    :param bird_2022: Audiodata of BirdClef2022
    :param bird_2021: Audiodata of BirdClef2021
    """

    meta_2024: pd.DataFrame
    meta_2023: pd.DataFrame | None = None
    meta_2022: pd.DataFrame | None = None
    meta_2021: pd.DataFrame | None = None
    bird_2024: npt.NDArray[Any] | None = None
    bird_2023: npt.NDArray[Any] | None = None
    bird_2022: npt.NDArray[Any] | None = None
    bird_2021: npt.NDArray[Any] | None = None

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        return "XData"

    def __getitem__(self, key) -> "XData":
        sliced_meta_2024 = self.meta_2024.iloc[key]
        sliced_bird_2024 = self.bird_2024[key]
        sliced_meta_2023 = self.meta_2023.iloc[key]
        sliced_bird_2023 = self.bird_2023[key]
        sliced_meta_2022 = self.meta_2022.iloc[key]
        sliced_bird_2022 = self.bird_2022[key]
        sliced_meta_2021 = self.meta_2021.iloc[key]
        sliced_bird_2021 = self.bird_2021[key]
        return XData(
            meta_2024=sliced_meta_2024,
            meta_2023=sliced_meta_2023,
            meta_2022=sliced_meta_2022,
            meta_2021=sliced_meta_2021,
            bird_2024=sliced_bird_2024,
            bird_2023=sliced_bird_2023,
            bird_2022=sliced_bird_2022,
            bird_2021=sliced_bird_2021
        )


