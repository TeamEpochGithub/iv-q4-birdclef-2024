"""Butter filter for eeg signals."""
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy import typing as npt
from scipy.signal import butter, lfilter
from tqdm import tqdm

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock
from src.typing.typing import XData


@dataclass
class ButterFilter(VerboseTransformationBlock):
    """Butter filter for eeg signals.

    :param lower: The lower bound of the filter, if 0, uses a low pass filter
    :param upper: The upper bound of the filter
    :param order: The order of the filter
    :param sampling_rate: The sampling rate
    """

    lower: float = 0
    upper: float = 20
    order: int = 2
    sampling_rate: float = 200
    method: str = "filter"
    ranges: list[list[float]] | None = None

    def custom_transform(self, data: XData, **kwargs: Any) -> XData:
        """Filter the eeg signals with a butter filter.

        :param data: The X data to transform, as tuple (eeg, spec, meta)
        :return: The transformed data
        """
        if self.method == "filter":
            return self.apply_filter(data)
        if self.method == "extend":
            return self.extend(data)
        raise ValueError(f"Method {self.method} not recognized")

    def apply_filter(self, data: XData) -> XData:
        """Filter the data with a butter filter.

        :param data: The data to filter
        :return: The filtered data
        """
        eeg = data.eeg
        if eeg is None:
            raise ValueError("No EEG data to transform")
        for key in tqdm(eeg.keys(), desc="Butter Filtering EEG data"):
            if self.lower == 0:
                # low pass
                eeg[key] = eeg[key].apply(self.butter_lowpass_filter)
            else:
                # bandpass
                eeg[key] = eeg[key].apply(self.butter_bandpass_filter)
        return data

    def extend(self, data: XData) -> XData:
        """Filter the data with a butter filter.

        :param data: The data to filter
        :return: The filtered data
        """
        eeg = data.eeg
        if eeg is None:
            raise ValueError("No EEG data to transform")
        if self.ranges is None:
            raise ValueError("No ranges provided")
        for key in tqdm(eeg.keys(), desc="Butter Filtering EEG data"):
            extended = []
            for lower, upper in self.ranges:
                self.lower = lower
                self.upper = upper
                curr_range = eeg[key].apply(self.butter_bandpass_filter)
                extended.append(curr_range)
                # Rename the columns of curr_range based on lower and upper
                for col in curr_range.columns:
                    curr_range.rename(columns={col: f"{col}_{lower}-{upper}"}, inplace=True)  # noqa: PD002
            eeg[key] = pd.concat(extended, axis=1)
        return data

    def butter_lowpass_filter(self, data: pd.DataFrame) -> npt.NDArray[np.float32]:
        """Filter the data with a butter filter.

        Taken from "https://www.kaggle.com/code/nartaa/features-head-starter.
        :param data: The data to filter
        :param cutoff_freq: The cutoff frequency
        :param sampling_rate: The sampling rate
        :param order: The order of the filter
        """
        nyquist = 0.5 * self.sampling_rate
        normal_cutoff = self.upper / nyquist
        b, a = butter(self.order, normal_cutoff, btype="low", analog=False, output="ba")
        return lfilter(b, a, data, axis=0).astype(np.float32)

    def butter_highpass_filter(self, data: pd.DataFrame) -> npt.NDArray[np.float32]:
        """Filter the data with a butter filter.

        Taken from "https://www.kaggle.com/code/nartaa/features-head-starter.
        :param data: The data to filter
        :param cutoff_freq: The cutoff frequency
        :param sampling_rate: The sampling rate
        :param order: The order of the filter
        """
        nyquist = 0.5 * self.sampling_rate
        normal_cutoff = self.lower / nyquist
        b, a = butter(self.order, normal_cutoff, btype="high", analog=False, output="ba")
        return lfilter(b, a, data, axis=0).astype(np.float32)

    def butter_bandpass_filter(self, eeg: pd.DataFrame) -> npt.NDArray[np.float32]:
        """Filter the data with a butter filter.

        :param eeg: The data to filter
        :return: The filtered data
        """
        b, a = butter(self.order, [self.lower, self.upper], fs=self.sampling_rate, btype="band")
        return lfilter(b, a, eeg).astype(np.float32)


if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    from src.modules.transformation.verbose_transformation_block import XData

    file = "data/raw/"
