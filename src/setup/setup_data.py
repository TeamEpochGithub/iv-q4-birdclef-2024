"""File containing all methods related to processing and formatting data before it is inserted into a pipeline.

- Since these methods are very competition specific none have been implemented here yet.
- Usually you'll have one data setup for training and another for making submissions.
"""
from typing import Any

import pandas as pd

from src.typing.typing import XData


def setup_train_x_data(path: str) -> Any:  # noqa: ANN401
    """Create train x data for pipeline.

    :param path: Metadata path
    :param cache_path: Path to save the cache

    :return: x data
    """
    # if isinstance(cache_path, str):
    #     cache_path = Path(cache_path)
    #
    # if cache_path is not None and os.path.exists(cache_path / "eeg_cache.pkl"):
    #     logger.info(f"Found pickle cache for Audio data at: {cache_path / 'eeg_cache.pkl'}")
    #     with open(cache_path / "audio_cache.pkl", "rb") as f:
    #         audio_data = pickle.load(f)
    #     logger.info("Loaded pickle cache for Audio data")
    #
    # else:
    #     # Load all .ogg files from the path also containing in recursive folders
    #     files_names = []
    #     audio_data = []
    #     all_dirs = sorted([x[0] for x in os.walk(path)])
    #     for subdir in tqdm(all_dirs, desc="Loading files"):
    #         for file in sorted(os.listdir(subdir)):
    #             if file.endswith(".ogg"):
    #                 file_path = os.path.join(subdir, file)
    #                 files_names.append(file_path)
    #                 audio_data.append(librosa.load(file_path, sr=32000, dtype=np.float32)[0])
    #
    #     if cache_path is not None:
    #         logger.info("Saving pickle cache for Audio data")
    #         with open(cache_path / "audio_cache.pkl", "wb") as f:
    #             pickle.dump(audio_data, f)
    #         logger.info(f"Saved pickle cache for Audio data at: {cache_path / 'audio_cache.pkl'}")
    #
    # return audio_data

    # Load the dataframe from the path

    metadata = pd.read_csv(path)
    metadata['samplename'] = metadata.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])

    return XData(meta_2024=metadata)


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
