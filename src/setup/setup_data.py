"""File containing all methods related to processing and formatting data before it is inserted into a pipeline.

- Since these methods are very competition specific none have been implemented here yet.
- Usually you'll have one data setup for training and another for making submissions.
"""
from typing import Any

import librosa
import numpy as np
import numpy.typing as npt
import pandas as pd
from dask import delayed

from src.typing.typing import XData


def setup_train_x_data(raw_path: str, path_2024: str) -> Any:  # noqa: ANN401
    """Create train x data for pipeline.

    :param raw_path: Raw path
    :param path_2024: Metadata path
    :param cache_path: Path to save the cache

    :return: x data
    """
    # TODO Load dataset from previous years (Bird 2023 and earlier)

    # Load the dataframe from the path (Bird 2024)
    metadata_2024 = pd.read_csv(path_2024)
    metadata_2024['samplename'] = metadata_2024.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])

    # Load the bird_2024 data
    filenames_2024 = metadata_2024.filename
    filenames_2024 = [raw_path + filename for filename in filenames_2024]

    bird_2024 = []
    # Load the bird_2024 data lazily
    for i in range(len(filenames_2024)):
        bird_2024.append(load_audio(filenames_2024[i]))

    bird_2024 = np.array(bird_2024)

    return XData(meta_2024=metadata_2024, bird_2024=bird_2024)


@delayed
def load_audio(path: str) -> npt.NDArray[np.float32]:
    """Load audio data lazily using librosa"""
    return librosa.load(path, sr=32000, dtype=np.float32)[0]


def setup_train_y_data(path: str) -> Any:  # noqa: ANN401
    """Create train y data for pipeline.

    :param path: Usually raw path is a parameter
    :return: y data
    """
    metadata = pd.read_csv(path)
    metadata['samplename'] = metadata.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])
    return metadata


def setup_inference_data(path: str) -> Any:  # noqa: ANN401
    """Create data for inference with pipeline.

    :param path: Usually raw path is a parameter
    :return: Inference data
    """
    return setup_train_x_data(path)


def setup_splitter_data() -> Any:  # noqa: ANN401
    """Create data for splitter.

    :return: Splitter data
    """
    return None
