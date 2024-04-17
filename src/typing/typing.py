"""Common type definitions for the project."""
from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch


@dataclass
class XData:
    """The X data to be used in the pipeline.

    :param eeg: The EEG data, as a dictionary of DataFrames
    :param kaggle_spec: The Kaggle spectrogram data, as a dictionary of Tensors
    :param eeg_spec: The EEG spectrogram data, as a dictionary of Tensors
    :param meta: The metadata, as a DataFrame
    :param shared: The shared data to be used in the pipeline. Contains frequency data, offset data, etc.
    :param features: Contains the features extracted from the EEG data. Should have same length as meta.
    """

    eeg: dict[int, pd.DataFrame] | None
    kaggle_spec: dict[int, torch.Tensor] | None
    eeg_spec: dict[int, torch.Tensor] | None
    meta: pd.DataFrame
    shared: dict[str, Any] | None
    features: pd.DataFrame | None = None

    def __getitem__(self, key: slice | int | list[int]) -> "XData":
        """Enable slice indexing on the meta attribute using iloc and filters other attributes based on eeg_id."""
        sliced_meta = self.meta.iloc[key]
        if isinstance(sliced_meta, pd.Series):
            sliced_meta = sliced_meta.to_frame()

        if self.features is not None:
            sliced_features = self.features.iloc[key]
            if isinstance(sliced_features, pd.Series):
                sliced_features = sliced_features.to_frame()
        else:
            sliced_features = None

        return XData(eeg=self.eeg, kaggle_spec=self.kaggle_spec, eeg_spec=self.eeg_spec, meta=sliced_meta, shared=self.shared, features=sliced_features)  # type: ignore[arg-type]

    def __len__(self) -> int:
        """Return the length of the meta attribute."""
        return len(self.meta)

    def __repr__(self) -> str:
        """Return a string representation of the object."""
