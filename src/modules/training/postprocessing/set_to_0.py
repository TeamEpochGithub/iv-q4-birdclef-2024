"""Set predictions to 0, if the model thinks with low probability that there is not a bird."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from src.modules.training.verbose_training_block import VerboseTrainingBlock


@dataclass
class SetTo0(VerboseTrainingBlock):
    """Set predictions to 0, if the model thinks with low probability that there is not a bird.

    :param consider_thresh: The threshold for considering a prediction as a bird.
    :param to_0_thresh: The threshold for setting a prediction to 0.
    """

    consider_thresh: float = 0.2
    to_0_thresh: float = 0.1

    def custom_train(self, x: npt.NDArray[np.float32], y: npt.NDArray[np.float32], **train_args: Any) -> tuple[Any, Any]:
        """Return the input data and labels."""
        return x, y

    def custom_predict(self, x: npt.NDArray[np.float32], **pred_args: Any) -> npt.NDArray[np.float32]:
        """Apply setting 0 to the predictions.

        :param x: The input data in shape (n_samples=48n, n_features=182)
        :return: The predictions
        """
        # Get the indices of the rows that sum < consider_thresh
        indices = np.where(x.sum(axis=1) < self.consider_thresh)[0]

        # Set the specific columns at the indices to 0 if the value is < to_0_thresh
        for i in tqdm(indices, desc="Setting to 0"):
            x[i, x[i] < self.to_0_thresh] = 0

        return x


# if __name__ == "__main__":
#     # Test Set0 block
#     set_to_0 = SetTo0()
#
#     x = np.array([[0.0001, 0.01, 0.05], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
#     x = set_to_0.custom_predict(x)
#     x_true = np.array([[0, 0.01, 0.05], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
#     print(x)
#     assert np.allclose(x, x_true)
