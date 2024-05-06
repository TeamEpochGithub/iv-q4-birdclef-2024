"""Butter filter for eeg signals."""
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from dask import delayed
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

    years: list[str] = field(default_factory=lambda: ["2024"])
    lower: float = 1250.0
    upper: float = 20000.0
    order: int = 5
    sampling_rate: float = 32000.0
    ranges: list[list[float]] | None = None

    def __post_init__(self) -> None:
        """Calculate the normal cutoff frequency."""
        nyquist = 0.5 * self.sampling_rate
        self.normal_cutoff = self.lower / nyquist

    def custom_transform(self, data: XData, **kwargs: Any) -> XData:
        """Filter the audio signals with a butter filter.

        :param data: The X data to transform (bird)
        :return: The transformed data
        """
        for year in self.years:
            attribute = f"bird_{year}"
            # Check if the attribute exists and is not None
            if hasattr(data, attribute) and getattr(data, attribute) is not None:
                curr_data = getattr(data, attribute)
                for i in tqdm(range(len(curr_data)), desc=f"Transforming {attribute} to butter filter"):
                    curr_data[i] = self.butter_highpass_filter(curr_data[i])
        return data

    def __call__(self, data: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Filter the data with a butter filter.

        :param data: The data to filter
        :return: The filtered data
        """
        return self.butter_highpass_filter(data)

    @delayed
    def butter_lowpass_filter(self, data: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Filter the data with a butter filter.

        Taken from "https://www.kaggle.com/code/nartaa/features-head-starter.
        :param data: The data to filter
        :param cutoff_freq: The cutoff frequency
        :param sampling_rate: The sampling rate
        :param order: The order of the filter
        """
        b, a = butter(self.order, self.normal_cutoff, btype="low", analog=False, output="ba")
        return lfilter(b, a, data, axis=0).astype(np.float32)

    @delayed
    def butter_highpass_filter(self, data: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Filter the data with a butter filter.

        Taken from "https://www.kaggle.com/code/nartaa/features-head-starter.
        :param data: The data to filter
        :param cutoff_freq: The cutoff frequency
        :param sampling_rate: The sampling rate
        :param order: The order of the filter
        """
        b, a = butter(self.order, self.normal_cutoff, btype="high", analog=False, output="ba")
        return lfilter(b, a, data, axis=0).astype(np.float32)

    def butter_bandpass_filter(self, eeg: pd.DataFrame) -> npt.NDArray[np.float32]:
        """Filter the data with a butter filter.

        :param eeg: The data to filter
        :return: The filtered data
        """
        b, a = butter(self.order, [self.lower, self.upper], fs=self.sampling_rate, btype="band")
        return lfilter(b, a, eeg).astype(np.float32)
