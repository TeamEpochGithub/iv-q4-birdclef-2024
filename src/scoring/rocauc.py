"""ROC AUC scorer from Kaggle."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score  # type: ignore[import-not-found]

from src.scoring.scorer import Scorer


@dataclass
class ROCAUC(Scorer):
    """OC AUC scorer from Kaggle."""

    name: str = "roc_auc"
    grade_threshold: float | None = None
    only_primary: bool = False

    def __call__(self, y_true: pd.DataFrame, y_pred: np.ndarray[Any, Any], **kwargs: Any) -> float:
        """Calculate the ROC AUC score.

        :param y_true: The true labels.
        :param y_pred: The predicted labels.

        :return: The ROC AUC score.
        """
        # Get metadata from the keyword arguments if not None

        metadata = kwargs.get("metadata")
        output_dir = kwargs.get("output")

        # Check if metadata is not None
        if metadata is None:
            raise ValueError("Metadata is required for this scorer.")

        # Convert both solution and submission to a dataframe
        if self.grade_threshold is not None:
            # Create a binary mask based on the rating column
            indices = metadata["rating"] >= self.grade_threshold
            # Use the mask to index the labels from y_true
            y_true = y_true[indices]

            # Also slice metadata
            metadata = metadata[indices]

        if self.only_primary:
            # Get the indices from the metadata where secondary label is an dempty list as string
            indices = metadata["secondary_labels"] == "[]"
            # Use the mask to index the labels from y_true
            y_true = y_true[indices]
            y_pred = y_pred[indices]

        # Convert
        solution = y_true
        submission = pd.DataFrame(np.clip(y_pred, 0, 1), columns=solution.columns)

        if not pd.api.types.is_numeric_dtype(submission.values):
            bad_dtypes = {x: submission[x].dtype for x in submission.columns if not pd.api.types.is_numeric_dtype(submission[x])}
            raise ValueError(f"Columns {bad_dtypes} have non-numeric dtypes.")

        solution_sums = solution.sum(axis=0)
        scored_columns = list(solution_sums[solution_sums > 0].index.values)
        # Raise an error if scored columns  <= 0
        if len(scored_columns) <= 0:
            raise ValueError("No positive labels in y_true, ROC AUC score is not defined in that case.")

        avg_roc_auc = roc_auc_score(solution[scored_columns].values, submission[scored_columns].values, average="macro")

        # Plot the ROC AUC score for each class and save the plot
        self.plot_class_scores(metadata, solution, submission, scored_columns, output_dir)

        # Calculate the ROC AUC score
        return avg_roc_auc

    def __str__(self) -> str:
        """Return the name of the scorer."""
        return self.name

    def plot_class_scores(self, metadata: pd.DataFrame, solution: pd.DataFrame, submission: pd.DataFrame, scored_columns: list[str], output_dir: str) -> None:  # type: ignore[arg-type]
        """Plot the ROC AUC score for each class and save the plot.

        :param metadata: The metadata.
        :param solution: The true labels.
        :param submission: The predicted labels.
        :param scored_columns: The scored columns.
        :param output_dir: The output directory.
        """
        # Calculate the ROC AUC score for each class

        class_roc_auc = roc_auc_score(solution[scored_columns].values, submission[scored_columns].values, average=None)

        fig, ax = plt.subplots(1, 1, figsize=(20, 60))

        # Plot using seaborn
        import seaborn as sns

        # Sort the scores and the columns
        # Map the scored columns to Scientific names in metadata
        scored_columns = [metadata[metadata["primary_label"] == x]["scientific_name"].to_numpy()[0] for x in scored_columns]

        # Get the count of the scored columns from the metadata
        scored_columns = [f"{x} ({metadata[metadata['scientific_name'] == x].shape[0]})" for x in scored_columns]

        scored_columns = [x for _, x in sorted(zip(class_roc_auc, scored_columns, strict=False), reverse=True)]
        class_roc_auc = sorted(class_roc_auc, reverse=True)

        # Score using seaborn also annotate the scores
        sns.barplot(x=class_roc_auc, y=scored_columns, ax=ax)
        for i, v in enumerate(class_roc_auc):
            ax.text(v + 0.01, i, str(round(v, 2)), color="black", va="center")
        ax.set_xlabel("ROC AUC Score")
        ax.set_ylabel("Bird Species")
        ax.set_title("ROC AUC Score for each Bird Species")

        # Save plot to output directory
        plt.savefig(f"{output_dir}/roc_auc_score.png")


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
