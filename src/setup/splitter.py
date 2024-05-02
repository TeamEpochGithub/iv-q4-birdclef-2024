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

    def split(self, data: Any, labels: Any) -> tuple[dict[int, Any], dict[int, Any]]:
        train_indices = {}
        test_indices = {}
        for year in self.years:
            train_indices[year] = []
            test_indices[year] = []
            for train_idx, test_idx in self.instantiated_splitter.split(data, labels):
                train_indices[year].append(train_idx)
                test_indices[year].append(test_idx)

        return train_indices, test_indices
