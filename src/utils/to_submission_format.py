"""File contains the function to_submission_format() which converts the output of the model to the submission format."""

import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.utils.logger import logger


def to_submission_format(predictions: npt.NDArray[np.float32], test_path: str | os.PathLike[str], species_path: str | os.PathLike[str]) -> pd.DataFrame:
    """Convert the predictions to the submission format.

    :param predictions: The predictions of the model.
    :param test_path: The path to the test data.
    :param species_path: The path to the species data.
    :raise ValueError: If the number of species in predictions does not match the number of species in the dataset.
    :return: The predictions in the submission format.
    """
    file_list = [file.stem for file in Path(test_path).glob("*.ogg")] + [file.stem for file in Path(test_path).glob("*.wav")]

    logger.info(f"Number of test soundscapes: {len(file_list)} ")
    logger.info(f"Filenames: {file_list[:10]}...")

    species_list = sorted(os.listdir(species_path))

    if np.isnan(predictions).any():
        logger.warning("Predictions contain NaN values. This is likely the result of a timeout. Replacing with zeros.")
        predictions = np.nan_to_num(predictions)

    if predictions.shape[1] != len(species_list):
        raise ValueError("Number of species in predictions does not match the number of species in the dataset.")

    if predictions.shape[0] != len(file_list) * 48:
        logger.warning(
            f"Number of predictions ({predictions.shape[0]}) does not match the number of test soundscapes ({len(file_list) * 48}). "
            "This is may be the result of a timeout. Padding submission with zeros.",
        )
        predictions = np.concatenate([predictions, np.zeros((len(file_list) * 48 - predictions.shape[0], predictions.shape[1]))])

    # Convert predictions to dataframe with species as columns and row_id as index.
    submission = pd.DataFrame(predictions, columns=species_list)
    submission["row_id"] = [f"{file}_{(i + 1) * 5}" for file in file_list for i in range(48)]
    return submission


# if __name__ == "__main__":
#     test_path = "data/test_soundscapes"
#     species_path = "data/species"
#     predictions = np.random.rand(48 * 7, 182)
#     to_submission_format(predictions, "../../data/raw/test_soundscapes", "../../data/raw/train_audio")
#     print("Test passed.")
