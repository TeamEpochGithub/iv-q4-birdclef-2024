"""ROC AUC scorer from Kaggle."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score  # type: ignore[import-not-found]

from src.scoring.scorer import Scorer
from src.typing.typing import YData


@dataclass
class ROCAUC(Scorer):
    """OC AUC scorer from Kaggle."""

    name: str = "roc_auc"
    grade_threshold: float | None = None
    only_primary: bool = False

    def __call__(self, y_true: YData, y_pred: np.ndarray[Any, Any], **kwargs: Any) -> float:
        """Calculate the ROC AUC score.

        :param y_true: The true labels.
        :param y_pred: The predicted labels.

        :return: The ROC AUC score.
        """
        # Get metadata from the keyword arguments if not None
        scores = {}
        # separate the preds for the years
        year_preds = {}
        start_idx = 0

        # Create union metadata
        label_lookup = pd.concat([y_true[f'label_{year}'] for year in kwargs.get('years')]).fillna(0).reset_index(drop=True)

        # Do the year splitting the same way as in XData and YData
        for year in sorted(kwargs.get('years'), reverse=True):
            year_preds[f'{year}'] = y_pred[start_idx:start_idx+len(kwargs.get('test_indices')[str(year)])]
            start_idx += len(kwargs.get('test_indices')[str(year)])

        # Loop over the 
        for year in sorted(kwargs.get('years'), reverse=True):
            metadata = y_true[f'meta_{year}'].iloc[kwargs.get('test_indices')[str(year)]]
            y_true_year = y_true[f'label_{year}'].iloc[kwargs.get('test_indices')[str(year)]]
            # Check if metadata is not None
            if metadata is None:
                raise ValueError("Metadata is required for this scorer.")

            # Convert both solution and submission to a dataframe
            if self.grade_threshold is not None:

                # Create a binary mask based on the rating column
                
                indices = metadata["rating"] >= self.grade_threshold
                # Use the mask to index the labels from y_true
                y_true_year = y_true_year[indices]

                # Also slice metadata
                metadata = metadata[indices]

            if self.only_primary:
                # Get the indices from the metadata where secondary label is an empty list as string
                indices = metadata["secondary_labels"] == "[]"
                # Use the mask to index the labels from y_true
                y_true_year = y_true_year[indices]
                # from the preds index the years data, then index that year using indices
                y_pred_year = year_preds[f'{year}'][indices]

            # Convert
            solution = y_true_year
            # Select the correct columns from the pred using the label_lookup
            label_indices = [label_lookup.columns.get_loc(col) for col in y_true_year.columns]

            submission = pd.DataFrame(np.clip(y_pred_year[:,label_indices], 0, 1), columns=solution.columns)

            if not pd.api.types.is_numeric_dtype(submission.values):
                bad_dtypes = {x: submission[x].dtype for x in submission.columns if not pd.api.types.is_numeric_dtype(submission[x])}
                raise ValueError(f"Columns {bad_dtypes} have non-numeric dtypes.")

            solution_sums = solution.sum(axis=0)
            scored_columns = list(solution_sums[solution_sums > 0].index.values)
            # Raise an error if scored columns  <= 0
            if len(scored_columns) <= 0:
                raise ValueError("No positive labels in y_true, ROC AUC score is not defined in that case.")
            scores[year] = roc_auc_score(solution[scored_columns].values, submission[scored_columns].values, average="macro")
        # Calculate the ROC AUC score
        return scores

    def __str__(self) -> str:
        """Return the name of the scorer."""
        return self.name


# Test the scorer
# if __name__ == "__main__":
#     scorer = ROCAUC("roc_auc")
#
#     birds = ["Sparrow", "Robin", "Crow", "Cardinal", "Hawk"]
#     predicted = [
#         [0.62, 0.21, 0.30, 0.04, 0.13],  # Sparrow, Crow
#         [0.22, 0.41, 0.26, 0.51, 0.11],  # Robin (False Cardinal now 0.51)
#         [0.82, 0.20, 0.93, 0.63, 0.13],  # Sparrow, Crow, Cardinal
#         [0.43, 0.10, 0.11, 0.48, 0.16],    # Sparrow, Cardinal (now 0.48)
#         [0.43, 0.10, 0.11, 0.48, 0.16],
#
#     ]
#     solution = [
#         [1, 0, 1, 0, 0],  # Sparrow, Crow
#         [0, 1, 0, 0, 0],  # Robin
#         [1, 0, 1, 1, 0],  # Sparrow, Crow, Cardinal
#         [1, 0, 0, 1, 0],
#         [0,0,0,0,0]# Sparrow, Cardinal
#     ]
#     solution_pd = pd.DataFrame(solution, columns=birds).astype(np.float32)
#     predicted_pd = np.array(predicted)
#     print("Score:", scorer(solution_pd, predicted_pd))
