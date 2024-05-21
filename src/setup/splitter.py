"""Wrapper class fo sklearn splitters that returns dicts."""

from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any

import sklearn.model_selection

from src.typing.typing import YData


@dataclass
class Splitter:
    """Wrapper class fo sklearn splitters that returns dicts.

    :param splitter: The sklearn splitter to use.
    :param n_splits: The number of splits to use.
    :param shuffle: Whether to shuffle the data.
    :param random_state: The random state to use.
    :param years: The years to split.
    """

    splitter: Any = sklearn.model_selection.StratifiedKFold
    n_splits: int = 5
    shuffle: bool = True
    random_state: int = 42
    years: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Fully instantiate the sklearn splitter."""
        # If empty list assume only 2024 is specified
        if len(self.years) == 0:
            self.years.append(2024)
        self.instantiated_splitter = self.splitter(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)

    def split(self, data: YData) -> Generator[list[list[dict[int, Any]]], list[list[dict[int, Any]]], None]:
        """Split the datasets for each year and yield the appropriate dicts.

        :param data: The YData object to split.
        :yields: The split datasets.
        """
        splits: list[list[dict[int, Any]]]
        splits = [[{}, {}] for _ in range(self.n_splits)]
        for year in self.years:
            year_splits = self.instantiated_splitter.split(data[f"meta_{year}"], data[f"meta_{year}"]["primary_label"])
            for i, split in enumerate(year_splits):
                splits[i][0][year] = split[0]
                splits[i][1][year] = split[1]

        for split in splits:
            yield split
