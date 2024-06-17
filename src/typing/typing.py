"""Common type definitions for the project."""

from __future__ import annotations

from collections.abc import Generator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias, cast, overload

import numpy as np
import numpy.typing as npt
import pandas as pd
from typing_extensions import Self

IlocIndexer: TypeAlias = int | slice | Sequence[int] | Sequence[bool]


@dataclass
class XData:
    """Dataclass to hold X data.

    :param meta_2024: Metadata of BirdClef2024
    :param meta_2024add: Additional metadata of BirdClef2024
    :param meta_2023: Metadata of BirdClef2023
    :param meta_2022: Metadata of BirdClef2022
    :param meta_2021: Metadata of BirdClef2021
    :param meta_kenya: Metadata of Kenya
    :param bird_2024: Audiodata of BirdClef2024
    :param bird_2024add: Additional audiodata of BirdClef2024
    :param bird_2023: Audiodata of BirdClef2023
    :param bird_2022: Audiodata of BirdClef2022
    :param bird_2021: Audiodata of BirdClef2021
    :param bird_kenya: Audiodata of Kenya
    """

    meta_2024: pd.DataFrame | None = None
    meta_2024add: pd.DataFrame | None = None
    meta_2024gsil: pd.DataFrame | None = None
    meta_2024gbird: pd.DataFrame | None = None
    meta_2024gxeno: pd.DataFrame | None = None
    meta_2023: pd.DataFrame | None = None
    meta_2022: pd.DataFrame | None = None
    meta_2021: pd.DataFrame | None = None
    meta_2020: pd.DataFrame | None = None
    meta_pam22: pd.DataFrame | None = None
    meta_pam21: pd.DataFrame | None = None
    meta_pam20: pd.DataFrame | None = None
    meta_kenya: pd.DataFrame | None = None
    meta_esc50: pd.DataFrame | None = None
    meta_green: pd.DataFrame | None = None
    meta_freefield: pd.DataFrame | None = None
    bird_2024: npt.NDArray[Any] | None = None
    bird_2024add: npt.NDArray[Any] | None = None
    bird_2024gsil: npt.NDArray[np.float32] | None = None
    bird_2024gbird: npt.NDArray[np.float32] | None = None
    bird_2024gxeno: npt.NDArray[np.float32] | None = None
    bird_2023: npt.NDArray[Any] | None = None
    bird_2022: npt.NDArray[Any] | None = None
    bird_2021: npt.NDArray[Any] | None = None
    bird_2020: npt.NDArray[Any] | None = None
    bird_pam22: npt.NDArray[np.float32] | None = None
    bird_pam21: npt.NDArray[np.float32] | None = None
    bird_pam20: npt.NDArray[np.float32] | None = None
    bird_kenya: npt.NDArray[Any] | None = None
    bird_esc50: npt.NDArray[Any] | None = None
    bird_green: npt.NDArray[Any] | None = None
    bird_freefield: npt.NDArray[np.float32] | None = None

    @overload
    def __getitem__(self, indexer: IlocIndexer) -> XData: ...

    @overload
    def __getitem__(self, indexer: Mapping[str, Any]) -> XData: ...

    @overload
    def __getitem__(self, indexer: str) -> pd.DataFrame | npt.NDArray[np.float32]: ...

    def __getitem__(self, indexer: IlocIndexer | Mapping[str, Any] | str) -> XData | pd.DataFrame | npt.NDArray[np.float32]:
        """Index the data according to the indexer type.

        :param indexer: The indexer to use
        :raise AttributeError: If trying to index non-existent 2024
        :return: The data requested
        """
        if isinstance(indexer, Mapping):
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

            if "union" in indexer:
                # Extract all the years to take the union of
                years = indexer.split("_")[2:]

                # Create the union field
                if indexer[:5] == "bird_" and not hasattr(self, indexer):
                    setattr(self, indexer, np.concatenate([self[f"bird_{year}"] for year in years]))
                if indexer[:5] == "meta_" and not hasattr(self, indexer):
                    setattr(self, indexer, pd.concat([cast(pd.DataFrame, self[f"meta_{year}"]) for year in years]).reset_index(drop=True))

            return getattr(self, indexer)

        # If nothing is specified assume we are using 2024 data
        if self.bird_2024 is None or self.meta_2024 is None:
            raise AttributeError("No data available for 2024")

        sliced_meta_2024 = pd.DataFrame(self.meta_2024.iloc[indexer])  # type: ignore[index]
        sliced_bird_2024 = self.bird_2024[indexer]

        return XData(
            meta_2024=sliced_meta_2024,
            bird_2024=sliced_bird_2024,
        )

    def __setitem__(self, key: str, value: pd.DataFrame | npt.NDArray[np.float32] | None) -> None:
        """Set the value of key.

        :param key: The key to set
        :param value: The value to set
        :raise KeyError: If the key is not a valid attribute of XData
        """
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise KeyError(f"'{key}' is not a valid attribute of {self}")

    def __delitem__(self, key: str) -> None:
        """Delete the value of key.

        :param key: The key to delete
        :raise KeyError: If the key is not a valid attribute of XData
        """
        if hasattr(self, key):
            setattr(self, key, None)
        else:
            raise KeyError(f"'{key}' is not a valid attribute of XData")

    def __repr__(self) -> str:
        """Return a string representation of the object.

        :return: The string representation
        """
        return "XData"

    @property
    def years(self) -> tuple[str, ...]:
        """The "years" present in the data."""
        return tuple(year.split("_")[1] for year in self.__dict__ if year[:5] == "bird_" and self[year] is not None)

    def __contains__(self, item: str) -> bool:
        """Check if the item is present in the data.

        :param item: The item to check
        :return: Whether the item is present
        """
        return item in self.__dict__ and self[item] is not None

    def __len__(self) -> int:
        """Get the total number of sounds in the data across all years.

        :return: The total number of sounds
        """
        return sum([len(self[f"bird_{year}"]) for year in self.years])

    def __bool__(self) -> bool:
        """Check if there is any data present.

        :return: Whether there is any data present
        """
        return any(self[f"bird_{year}"] is not None for year in self.years)

    def __iter__(self) -> Self:
        """Iterate over itself.

        :return: Self
        """
        return self

    def __next__(self) -> Generator[tuple[str, pd.DataFrame, npt.NDArray[np.float32]], None, None]:
        """Get the next year's data.

        :return: The year, metadata, and audiodata
        """
        for year in self.years:
            yield year, cast(pd.DataFrame, self[f"meta_{year}"]), cast(npt.NDArray[np.float32], self[f"bird_{year}"])


@dataclass
class YData:
    """Dataclass to hold Y data.

    :param meta_2024: Metadata of BirdClef2024
    :param meta_2024add: Additional metadata of BirdClef2024
    :param meta_2023: Metadata of BirdClef2023
    :param meta_2022: Metadata of BirdClef2022
    :param meta_2021: Metadata of BirdClef2021
    :param meta_kenya: Metadata of Kenya
    :param label_2024: Labels of BirdClef2024
    :param label_2024add: Additional labels of BirdClef2024
    :param label_2023: Labels of BirdClef2023
    :param label_2022: Labels of BirdClef2022
    :param label_2021: Labels of BirdClef2021
    :param label_kenya: Labels of Kenya
    """

    meta_2024: pd.DataFrame | None = None
    meta_2024add: pd.DataFrame | None = None
    meta_2024gsil: pd.DataFrame | None = None
    meta_2024gbird: pd.DataFrame | None = None
    meta_2024gxeno: pd.DataFrame | None = None
    meta_2023: pd.DataFrame | None = None
    meta_2022: pd.DataFrame | None = None
    meta_2021: pd.DataFrame | None = None
    meta_2020: pd.DataFrame | None = None
    meta_pam22: pd.DataFrame | None = None
    meta_pam21: pd.DataFrame | None = None
    meta_pam20: pd.DataFrame | None = None
    meta_kenya: pd.DataFrame | None = None
    meta_esc50: pd.DataFrame | None = None
    meta_green: pd.DataFrame | None = None
    meta_freefield: pd.DataFrame | None = None
    label_2024: pd.DataFrame | None = None
    label_2024add: pd.DataFrame | None = None
    label_2024gsil: pd.DataFrame | None = None
    label_2024gbird: pd.DataFrame | None = None
    label_2024gxeno: pd.DataFrame | None = None
    label_2023: pd.DataFrame | None = None
    label_2022: pd.DataFrame | None = None
    label_2021: pd.DataFrame | None = None
    label_2020: pd.DataFrame | None = None
    label_pam22: pd.DataFrame | None = None
    label_pam21: pd.DataFrame | None = None
    label_pam20: pd.DataFrame | None = None
    label_kenya: pd.DataFrame | None = None
    label_esc50: pd.DataFrame | None = None
    label_green: pd.DataFrame | None = None
    label_freefield: pd.DataFrame | None = None

    @overload
    def __getitem__(self, indexer: IlocIndexer) -> YData: ...

    @overload
    def __getitem__(self, indexer: Mapping[str, Any]) -> YData: ...

    @overload
    def __getitem__(self, indexer: str) -> pd.DataFrame: ...

    def __getitem__(self, indexer: IlocIndexer | Mapping[str, Any] | str) -> YData | pd.DataFrame:
        """Index the data according to the indexer type.

        :param indexer: The indexer to use
        :raise AttributeError: If trying to index non-existent 2024
        :return: The data requested
        """
        if isinstance(indexer, Mapping):
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

            if "union" in indexer:
                # Extract all the years to take the union of
                years = indexer.split("_")[2:]

                # Create the union field
                if indexer[:6] == "label_" and not hasattr(self, indexer):
                    setattr(self, indexer, pd.concat([self[f"label_{year}"] for year in years]).fillna(0).reset_index(drop=True))
                if indexer[:5] == "meta_" and not hasattr(self, indexer):
                    setattr(self, indexer, pd.concat([self[f"meta_{year}"] for year in years]).reset_index(drop=True))

            return getattr(self, indexer)

        # If indices are not a dict assume that we are using the 2024 data
        if self.label_2024 is None or self.meta_2024 is None:
            raise AttributeError("No data available for 2024")

        sliced_meta_2024 = pd.DataFrame(self.meta_2024.iloc[indexer])  # type: ignore[index]
        sliced_label_2024 = pd.DataFrame(self.label_2024.iloc[indexer])  # type: ignore[index]

        return YData(
            meta_2024=sliced_meta_2024,
            label_2024=sliced_label_2024,
        )

    def __setitem__(self, key: str, value: pd.DataFrame | npt.NDArray[Any] | None) -> None:
        """Set the value of key.

        :param key: The key to set
        :param value: The value to set
        :raise KeyError: If the key is not a valid attribute of YData
        """
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise KeyError(f"'{key}' is not a valid attribute of {self}")

    def __delitem__(self, key: str) -> None:
        """Delete the value of key.

        :param key: The key to delete
        :raise KeyError: If the key is not a valid attribute of YData
        """
        if hasattr(self, key):
            setattr(self, key, None)
        else:
            raise KeyError(f"'{key}' is not a valid attribute of YData")

    def __repr__(self) -> str:
        """Return a string representation of the object.

        :return: The string representation
        """
        return "YData"

    @property
    def years(self) -> tuple[str, ...]:
        """Return the "years" present in the data."""
        return tuple(year.split("_")[1] for year in self.__dict__ if year[:6] == "label_" and self[year] is not None)

    def __contains__(self, item: str) -> bool:
        """Check if the item is present in the data.

        :param item: The item to check
        :return: Whether the item is present
        """
        return item in self.__dict__ and self[item] is not None

    def __len__(self) -> int:
        """Get the total number of sounds in the data across all years.

        :return: The total number of sounds
        """
        return sum([len(self[f"label_{year}"]) for year in self.years])

    def __bool__(self) -> bool:
        """Check if there is any data present.

        :return: Whether there is any data present
        """
        return any(self[f"label_{year}"] is not None for year in self.years)

    def __iter__(self) -> Self:
        """Iterate over itself.

        :return: Self
        """
        return self

    def __next__(self) -> Generator[tuple[str, pd.DataFrame, pd.DataFrame], None, None]:
        """Get the next year's data.

        :return: The year, metadata, and labels
        """
        for year in self.years:
            yield year, self[f"meta_{year}"], self[f"label_{year}"]
