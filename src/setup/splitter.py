from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any

import sklearn
import sklearn.model_selection


@dataclass
class Splitter:
    splitter: Any = sklearn.model_selection.StratifiedKFold
    n_splits: int = 5
    shuffle: bool = True
    random_state: int = 42
    years: list[int] = field(default_factory=list)

    def __post_init__(self):
        # If empty list assume only 2024 is specified
        if len(self.years) == 0:
            self.years.append(2024)
        self.instantiated_splitter = self.splitter(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)

    def split(self, data: Any, labels: Any) -> Generator[list[dict[int, Any], dict[int, Any]], list[dict[int, Any], dict[int, Any]], None]:
        # make an empty dict for each split
        splits: list[list[dict[int,Any], dict[int,Any]]]
        splits = [[{}, {}] for _ in range(5)]
        for year in self.years:
            year_splits = self.instantiated_splitter.split(data[f"meta_{year}"], data[f"meta_{year}"]["primary_label"])
            for i, split in enumerate(year_splits):
                splits[i][0][year] = split[0]
                splits[i][1][year] = split[1]

        for split in splits:
            yield split
