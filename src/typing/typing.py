"""Common type definitions for the project."""
from dataclasses import dataclass
from typing import Any

import numpy as np
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

    meta_2024: pd.DataFrame | None = None
    meta_2023: pd.DataFrame | None = None
    meta_2022: pd.DataFrame | None = None
    meta_2021: pd.DataFrame | None = None
    bird_2024: npt.NDArray[Any] | None = None
    bird_2023: npt.NDArray[Any] | None = None
    bird_2022: npt.NDArray[Any] | None = None
    bird_2021: npt.NDArray[Any] | None = None

    def __getitem__(self, indexer: Any) -> "XData" | pd.DataFrame | npt.NDArray[np.float32]:  # noqa: ANN401 C901
        """Index the data according to the indexer type."""
        if isinstance(indexer, dict):
            sliced_fields = {}
            # Slice all the years by the appropriate indices and save to a dict
            for year in indexer:
                if getattr(self, f"bird_{year}") is not None:
                    sliced_fields[f"bird_{year}"] = getattr(self, f"bird_{year}")[indexer[year]]
                if getattr(self, f"meta_{year}") is not None:
                    sliced_fields[f"meta_{year}"] = getattr(self, f"meta_{year}").iloc[indexer[year]]
            return XData(**sliced_fields)
        if isinstance(indexer, str):
            # allow dict like indexing with keys

            if "union" in indexer.split("_"):
                # Extract all the years to take the union of
                years = sorted([year for year in indexer.split("_") if year.isdigit()], reverse=True)  # Start from 2024

                # Create the union field
                if indexer[:5] == "bird_" and not hasattr(self, indexer):
                    setattr(self, indexer, np.concatenate([self[f"bird_{year}"] for year in years]))
                if indexer[:5] == "meta_" and not hasattr(self, indexer):
                    setattr(self, indexer, pd.concat([self[f"meta_{year}"] for year in years]).reset_index(drop=True))

            return getattr(self, indexer)

        # If nothing is specified assume we are using 2024 data
        if self.meta_2024 is not None:
            sliced_meta_2024 = self.meta_2024.iloc[indexer]
        if self.bird_2024 is not None:
            sliced_bird_2024 = self.bird_2024[indexer]

        return XData(
            meta_2024=sliced_meta_2024,
            bird_2024=sliced_bird_2024,
        )

    def __setitem__(self, key: str, value: pd.DataFrame | npt.NDArray[Any] | None) -> None:
        """Set the value of key."""
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise KeyError(f"'{key}' is not a valid attribute of XData")

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        return "XData"


@dataclass
class YData:
    """Dataclass to hold X data.

    :param meta_2024: Metadata of BirdClef2024:
    :param meta_2023: Metadata of BirdClef2023:
    :param meta_2022: Metadata of BirdClef2022:
    :param meta_2021: Metadata of BirdClef2021:
    :param label_2024: Labels of BirdClef2024
    :param label_2023: Labels of BirdClef2023
    :param label_2022: Labels of BirdClef2022
    :param label_2021: Labels of BirdClef2021
    """

    meta_2024: pd.DataFrame | None = None
    meta_2023: pd.DataFrame | None = None
    meta_2022: pd.DataFrame | None = None
    meta_2021: pd.DataFrame | None = None
    label_2024: pd.DataFrame | None = None
    label_2023: pd.DataFrame | None = None
    label_2022: pd.DataFrame | None = None
    label_2021: pd.DataFrame | None = None

    def __getitem__(self, indexer: Any) -> "YData" | pd.DataFrame:  # noqa: ANN401 C901
        """Index the data according to the indexer type."""
        if isinstance(indexer, dict):
            sliced_fileds = {}
            # Slice all the years by the appropriate indices and save to a dict
            for year in indexer:
                if getattr(self, f"label_{year}") is not None:
                    sliced_fileds[f"label_{year}"] = getattr(self, f"label_{year}").iloc[indexer[year]]
                if getattr(self, f"meta_{year}") is not None:
                    sliced_fileds[f"meta_{year}"] = getattr(self, f"meta_{year}").iloc[indexer[year]]
            return YData(**sliced_fileds)

        if isinstance(indexer, str):
            # allow dict like indexing with keys

            if "union" in indexer.split("_"):
                # Extract all the years to take the union of
                years = sorted([year for year in indexer.split("_") if year.isdigit()], reverse=True)  # Start from 2024

                # Create the union field
                if indexer[:6] == "label_" and not hasattr(self, indexer):
                    setattr(self, indexer, pd.concat([self[f"label_{year}"] for year in years]).fillna(0).reset_index(drop=True))
                if indexer[:5] == "meta_" and not hasattr(self, indexer):
                    setattr(self, indexer, pd.concat([self[f"meta_{year}"] for year in years]).reset_index(drop=True))

            return getattr(self, indexer)

        # If indices are not a dict assume that we are using the 2024 data
        if self.meta_2024 is not None:
            sliced_meta_2024 = self.meta_2024.iloc[indexer]
        if self.label_2024 is not None:
            sliced_label_2024 = self.label_2024.iloc[indexer]

        return YData(
            meta_2024=sliced_meta_2024,
            label_2024=sliced_label_2024,
        )

    def __setitem__(self, key: str, value: pd.DataFrame | npt.NDArray[Any] | None) -> None:
        """Set the value of key."""
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise KeyError(f"'{key}' is not a valid attribute of YData")

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        return "YData"
