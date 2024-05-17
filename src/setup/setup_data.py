"""File containing all methods related to processing and formatting data before it is inserted into a pipeline.

- Since these methods are very competition specific none have been implemented here yet.
- Usually you'll have one data setup for training and another for making submissions.
"""
import ast
import glob
from typing import Any

import librosa
import numpy as np
import numpy.typing as npt
import pandas as pd
from dask import delayed

from src.typing.typing import XData, YData
from src.utils.logger import logger


def setup_train_x_data(raw_path: str, years: list[int]) -> XData:
    """Create train x data for pipeline.

    :param raw_path: Raw path
    :param years: The years you want to use the data from

    :return: XData object
    """
    # Instantiate an empty XData object to later fill with data
    xdata = XData()

    for year in years:
        metadata_path = raw_path + str(year) + "/" + "train_metadata.csv"
        data_path = raw_path + str(year) + "/" + "train_audio/"
        metadata = pd.read_csv(metadata_path)
        metadata["samplename"] = metadata.filename.map(lambda x: x.split("/")[0] + "-" + x.split("/")[-1].split(".")[0])
        #metadata = metadata[metadata['secondary_labels'] == '[]'].reset_index(drop=True)

        # Load the bird_2024 data
        filenames = metadata.filename
        filenames = [data_path + filename for filename in filenames]

        bird = np.array([load_audio_train(filename) for filename in filenames])
        xdata[f"bird_{year}"] = bird
        xdata[f"meta_{year}"] = metadata

    return xdata


@delayed
def load_audio_train(path: str) -> npt.NDArray[np.float32]:
    """Load audio data lazily using librosa.

    :param path: Path to the audio file
    :return: Audio data
    """
    return librosa.load(path, sr=32000, dtype=np.float32)[0]


@delayed
def load_audio_submit(path: str) -> npt.NDArray[np.float32]:
    """Load audio data lazily using librosa.

    :param path: Path to the audio file
    :return: Audio data
    """
    return librosa.load(path, sr=32000, dtype=np.float32)[0] / 100


def setup_train_y_data(raw_path: str, years: list[str]) -> YData:
    """Create train y data for pipeline.

    :param raw_path: path to the raw data
    :param years: The years you want to use the data from

    :return: YData object
    """
    ydata = YData()

    for year in years:
        metadata_path = raw_path + str(year) + "/" + "train_metadata.csv"
        metadata = pd.read_csv(metadata_path)
        metadata["samplename"] = metadata.filename.map(lambda x: x.split("/")[0] + "-" + x.split("/")[-1].split(".")[0])
        #metadata = metadata[metadata['secondary_labels'] == '[]'].reset_index(drop=True)

        ydata[f"meta_{year}"] = metadata

        if "labels" in metadata.columns:
            ydata[f"label_{year}"] = one_hot_label(metadata)
        else:
            ydata[f"label_{year}"] = one_hot_primary_secondary(metadata)
    return ydata


def one_hot_primary_secondary(metadata: pd.DataFrame) -> pd.DataFrame:
    """Create one-hot encoded labels for primary and secondary labels.

    :param metadata: Metadata dataframe
    :return: One-hot encoded labels
    """
    # Get all unique primary labels
    primary_labels_dict = {bird: index for index, bird in enumerate(metadata.primary_label.unique())}

    # For each row in the metadata, create a one-hot encoded dataframe for the primary labels using dummies
    one_hot = pd.get_dummies(metadata.primary_label).astype(np.float32)

    errors = []
    for i, secondary_labels in enumerate(metadata.secondary_labels):
        if secondary_labels == "[]":
            continue
        for secondary_label in ast.literal_eval(secondary_labels):
            try:
                one_hot.iloc[i, primary_labels_dict[secondary_label]] = 0.0
            except KeyError:  # noqa: PERF203
                errors.append((i, secondary_label))
    logger.debug(f"Errors: {errors}")

    return one_hot


def one_hot_label(metadata: pd.DataFrame) -> pd.DataFrame:
    """Create one-hot encoded labels for labels lists.

    :param metadata: Metadata dataframe
    :return: One-hot encoded labels
    """
    # Get all unique species
    species = set()
    for labels in metadata.labels:
        species.update(ast.literal_eval(labels))

    # Create empty columns
    bird_cols = sorted(species)
    one_hot = pd.DataFrame(0, index=np.arange(len(metadata)), columns=bird_cols)

    # Fill in the one-hot encoded labels
    for i, labels in enumerate(metadata.labels):
        for label in ast.literal_eval(labels):
            one_hot.iloc[i, bird_cols.index(label)] = 1
    return one_hot


def setup_inference_data(path: str) -> Any:  # noqa: ANN401
    """Create data for inference with pipeline.

    :param raw_path: Raw path
    :param path: Usually raw path is a parameter
    :return: Inference data
    """
    # Load all files in the path that end with .ogg with glob
    filenames = glob.glob(path + "/*.ogg") + glob.glob(path + "/*.wav")

    logger.info(f"Filenames: {filenames[:10]}...")

    # Load the bird_2024 data
    bird_2024 = np.array([load_audio_submit(filename) for filename in filenames])

    return XData(bird_2024=bird_2024)


def setup_splitter_data() -> Any:  # noqa: ANN401
    """Create data for splitter.

    :return: Splitter data
    """
    return None
