"""ROC AUC scorer from Kaggle."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score  # type: ignore[import-not-found]

from src.scoring.scorer import Scorer


class ROCAUC(Scorer):
    """OC AUC scorer from Kaggle."""

    def __init__(self, name: str) -> None:
        """Initialize the scorer with a name."""
        self.name = name

    def __call__(self, y_true: np.ndarray[Any, Any], y_pred: np.ndarray[Any, Any], **kwargs: Any) -> float:
        """Calculate the ROC AUC score.

        :param y_true: The true labels.
        :param y_pred: The predicted labels.

        :return: The ROC AUC score.
        """
        # Convert both solution and submission to a dataframe
        solution = pd.DataFrame(y_true)
        submission = pd.DataFrame(y_pred)

        if not pd.api.types.is_numeric_dtype(submission.values):
            bad_dtypes = {x: submission[x].dtype for x in submission.columns if not pd.api.types.is_numeric_dtype(submission[x])}
            raise ValueError(f"Columns {bad_dtypes} have non-numeric dtypes.")

        solution_sums = solution.sum(axis=0)
        scored_columns = list(solution_sums[solution_sums > 0].index.values)
        # Raise an error if scored columns  <= 0
        if len(scored_columns) <= 0:
            raise ValueError("No positive labels in y_true, ROC AUC score is not defined in that case.")

        # Calculate the ROC AUC score
        return roc_auc_score(solution[scored_columns].values, submission[scored_columns].values, average="macro")

    def __str__(self) -> str:
        """Return the name of the scorer."""
        return self.name


# Test the scorer
# if __name__ == "__main__":
#     y_true = np.array([[1, 0, 1, 0, 1, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
#     y_pred = np.array([[1, 0, 1, 0, 1, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0]])
#     scorer = ROCAUC("roc_auc")
#     score = scorer(y_true, y_pred)
#     print(score)
#     print("Test passed.")
