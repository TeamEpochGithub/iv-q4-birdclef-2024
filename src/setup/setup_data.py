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


def setup_train_x_data(data_path: str, path_2024: str) -> Any:  # noqa: ANN401
    """Create train x data for pipeline.

    :param raw_path: Raw path
    :param path_2024: Metadata path
    :param cache_path: Path to save the cache

    :return: x data
    """
    # TODO(someone?): Load dataset from previous years (Bird 2023 and earlier)

    # Load the dataframe from the path (Bird 2024)
    metadata_2024 = pd.read_csv(path_2024)
    metadata_2024["samplename"] = metadata_2024.filename.map(lambda x: x.split("/")[0] + "-" + x.split("/")[-1].split(".")[0])

    # Load the bird_2024 data
    filenames_2024 = metadata_2024.filename
    filenames_2024 = [data_path + filename for filename in filenames_2024]

    bird_2024 = np.array([load_audio(filename) for filename in filenames_2024])

    return XData(meta_2024=metadata_2024, bird_2024=bird_2024)


@delayed
def load_audio(path: str) -> npt.NDArray[np.float32]:
    """Load audio data lazily using librosa.

    :param path: Path to the audio file
    :return: Audio data
    """
    return librosa.load(path, sr=32000, dtype=np.float32)[0] / 200


def setup_train_y_data(path: str) -> YData:
    """Create train y data for pipeline.

    :param path: Usually raw path is a parameter
    :return: YData object
    """
    metadata = pd.read_csv(path)
    metadata["samplename"] = metadata.filename.map(lambda x: x.split("/")[0] + "-" + x.split("/")[-1].split(".")[0])

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
                one_hot.iloc[i, primary_labels_dict[secondary_label]] = 0.5
            except KeyError:  # noqa: PERF203
                errors.append((i, secondary_label))

    logger.debug(f"Errors: {errors}")
    logger.warning("Some secondary labels were not found in the primary labels. Check the output directory for more information.")
    return YData(meta_2024=metadata, label_2024=one_hot)


def setup_inference_data(path: str) -> Any:  # noqa: ANN401
    """Create data for inference with pipeline.

    :param raw_path: Raw path
    :param path: Usually raw path is a parameter
    :return: Inference data
    """
    # Load all files in the path that end with .ogg with glob
    filenames = glob.glob(path + "/*.ogg")

    logger.info(f"Filenames: {filenames[:10]}...")

    # Load the bird_2024 data
    bird_2024 = np.array([load_audio(filename) for filename in filenames])

    return XData(bird_2024=bird_2024)


def setup_splitter_data() -> Any:  # noqa: ANN401
    """Create data for splitter.

    :return: Splitter data
    """
    return None
