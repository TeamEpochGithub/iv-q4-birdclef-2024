"""Common type definitions for the project."""
from dataclasses import dataclass
import dask.array as da

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
    bird_2024: dict[str, da.Array] | None = None
    bird_2023: dict[str, da.Array] | None = None
    bird_2022: dict[str, da.Array] | None = None
    bird_2021: dict[str, da.Array] | None = None

    def __len__(self) -> int:
        """Return the length of the meta attribute."""
        return len(self.meta_2024)

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        return "XData"
