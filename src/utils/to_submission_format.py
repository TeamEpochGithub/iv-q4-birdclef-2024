"""File contains the function to_submission_format() which converts the output of the model to the submission format."""
import glob
import os

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.utils.logger import logger


def to_submission_format(predictions: npt.NDArray[np.float32], test_path: str, species_path: str) -> pd.DataFrame:
    """Convert the predictions to the submission format.

    :param predictions (np.ndarray): The predictions of the model.
    :param test_path (str): The path to the test data.
    :param species_path (str): The path to the species data.

    :return: The predictions in the submission format.

    """
    file_list = glob.glob(test_path + "/*.ogg")
    file_list = [curr_file.split("/")[-1].split(".")[0] for curr_file in file_list]

    logger.info(f"Number of test soundscapes: {len(file_list)} ")
    logger.info(f"Filenames: {file_list[:10]}...")

    species_list = sorted(os.listdir(species_path))

    if predictions.shape[1] != len(species_list):
        raise ValueError("Number of species in predictions does not match the number of species in the dataset.")

    if predictions.shape[0] != len(file_list) * 48:
        raise ValueError("Number of predictions does not match the number of test soundscapes.")

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
