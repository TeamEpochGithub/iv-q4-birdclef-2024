"""ROC AUC scorer from Kaggle."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import roc_auc_score
from typing_extensions import override

from src.modules.logging.logger import logger
from src.scoring.scorer import Scorer
from src.typing.typing import YData


@dataclass
class ROCAUC(Scorer):
    """ROC AUC scorer from Kaggle.

    :param name: The name of the scorer.
    :param grade_threshold: The grade threshold.
    :param only_primary: Whether to only use the primary labels.
    """

    name: str = "roc_auc"
    grade_threshold: float | None = None
    only_primary: bool = False

    @override
    def __call__(self, y_true: YData, y_pred: npt.NDArray[Any], **kwargs: Any) -> dict[str, float]:
        """Calculate the ROC AUC score.

        :param y_true: The true labels.
        :param y_pred: The predicted labels.
        :param test_indices: The test indices.
        :param years: The datasets to use.
        :param output_dir: The output directory.
        :param kwargs: Additional keyword arguments.
        :raise ValueError: If metadata is required for this scorer, or if no positive labels are present in y_true, or if columns have non-numeric dtypes.
        :return: The ROC AUC score.
        """
        # Get metadata from the keyword arguments if not None
        scores: dict[str, float] = {}
        # separate the preds for the years
        year_preds: dict[str, npt.NDArray[Any]] = {}
        start_idx = 0

        # Retrieve the necessary keyword arguments
        test_indices: Mapping[str, Sequence[int]] = kwargs["test_indices"]
        years: Iterable[str] = kwargs["years"]

        # Create union metadata
        label_lookup = pd.concat([y_true[f"label_{year}"] for year in years]).fillna(0).reset_index(drop=True)

        # Do the year splitting the same way as in XData and YData
        for year in years:
            year_preds[str(year)] = y_pred[start_idx : start_idx + len(test_indices[str(year)])]
            start_idx += len(test_indices[str(year)])

        # Loop over the years
        for year in years:
            logger.info(f"Calculating ROC AUC for year {year}")

            metadata = y_true[f"meta_{year}"].iloc[test_indices[str(year)]]  # type: ignore[call-overload]
            y_true_year = y_true[f"label_{year}"].iloc[test_indices[str(year)]]  # type: ignore[call-overload]
            if y_true_year.sum().sum() == 0:
                logger.warning(f"No positive labels in y_true for year {year}, skipping ROC AUC calculation.")
                continue
            # Check if metadata is not None
            if metadata is None or y_true_year is None:
                raise ValueError("Metadata is required for this scorer.")

            # Convert both solution and submission to a dataframe
            if self.grade_threshold is not None:
                # Create a binary mask based on the rating column

                indices = metadata["rating"] >= self.grade_threshold
                # Use the mask to index the labels from y_true
                y_true_year = y_true_year[indices]

                # Also slice metadata
                metadata = metadata[indices]

            if self.only_primary and "secondary_labels" in metadata.columns and year not in ["kenya", "pam20", "pam21", "pam22"]:
                # Get the indices from the metadata where secondary label is an empty list as string
                indices = metadata["secondary_labels"] == "[]"

                # Use the mask to index the labels from y_true
                y_true_year = y_true_year[indices]
                # from the preds index the years data, then index that year using indices
                y_pred_year = year_preds[f"{year}"][indices]
            else:
                y_pred_year = year_preds[f"{year}"]

            # Convert
            solution = y_true_year
            # Select the correct columns from the pred using the label_lookup
            if not isinstance(y_true_year, pd.DataFrame):
                submission = pd.DataFrame(y_pred_year)
            else:
                label_indices = np.array([label_lookup.columns.get_loc(col) for col in y_true_year.columns])

                submission = pd.DataFrame(np.clip(y_pred_year[:, label_indices], 0, 1), columns=solution.columns)

            if not pd.api.types.is_numeric_dtype(submission.values):
                bad_dtypes = {x: submission[x].dtype for x in submission.columns if not pd.api.types.is_numeric_dtype(submission[x])}
                raise ValueError(f"Columns {bad_dtypes} have non-numeric dtypes.")

            solution_sums = solution.sum(axis=0)
            scored_columns = list(solution_sums[solution_sums > 0].index.values)
            # Raise an error if scored columns  <= 0
            if len(scored_columns) <= 0:
                raise ValueError("No positive labels in y_true, ROC AUC score is not defined in that case.")
            scores[year] = roc_auc_score(solution[scored_columns].values, submission[scored_columns].values, average="macro")
            # self.plot_class_scores(metadata=metadata, solution=solution, submission=submission, scored_columns=scored_columns, output_dir=output_dir, year=year)
        # Calculate the ROC AUC score
        return scores

    def plot_class_scores(self, metadata: pd.DataFrame, solution: pd.DataFrame, submission: pd.DataFrame, scored_columns: list[str], output_dir: str, year: str) -> None:
        """Plot the ROC AUC score for each class and save the plot.

        :param metadata: The metadata.
        :param solution: The true labels.
        :param submission: The predicted labels.
        :param scored_columns: The scored columns.
        :param output_dir: The output directory.
        :param year: The year.
        """
        # Calculate the ROC AUC score for each class
        class_roc_auc = roc_auc_score(solution[scored_columns].values, submission[scored_columns].values, average=None)

        fig, ax = plt.subplots(1, 1, figsize=(20, 60))

        # Plot using seaborn
        import seaborn as sns

        if "scientific_name" in metadata.columns:
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
        plt.savefig(f"{output_dir}/roc_auc_score_{year}.png")


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
