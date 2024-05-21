"""Visualize predictions block that creates a stacked line chart."""

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from typing_extensions import Never

from src.modules.training.verbose_training_block import VerboseTrainingBlock


@dataclass
class VisualizePreds(VerboseTrainingBlock):
    """Visualize predictions block that creates a stacked line chart.

    :param n: The number of 5-second intervals to visualize.
    :param threshold: The threshold for the top N lines.
    """

    n: int = 10
    threshold: float = 0.01

    def custom_train(
        self,
        x: npt.NDArray[np.floating[Any]],
        y: npt.NDArray[np.floating[Any]],
        **train_args: Never,
    ) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
        """Return the input data and labels.

        :param x: The input data in shape (n_samples=48n, n_features=182)
        :param y: The labels in shape (n_samples=48n, n_features=182)
        :return: The input data and labels
        """
        return x, y

    def custom_predict(self, x: npt.NDArray[np.floating[Any]], **pred_args: str) -> npt.NDArray[np.floating[Any]]:
        """Make multiple plots to visualize the predictions.

        :param x: The input data in shape (n_samples=48n, n_features=182)
        :param pred_args: The prediction arguments.
        :return: The predictions
        """
        # If on Kaggle, skip visualization
        if not torch.cuda.is_available():
            return x

        output_dir: str = pred_args["output_dir"]
        data_dir: str = pred_args["data_dir"]
        species_dir: str = pred_args["species_dir"]

        file_list = [file.stem for file in Path(data_dir).glob("*.ogg")]
        bird_classes = sorted(os.listdir(species_dir))

        # Create folders
        output_dir_heatmaps = Path(f"{output_dir}/heatmaps")
        output_dir_heatmaps.mkdir(parents=True, exist_ok=True)

        output_dir_top_n_lines = Path(f"{output_dir}/top_n_lines")
        output_dir_top_n_lines.mkdir(parents=True, exist_ok=True)

        for i, sliced in tqdm(enumerate(range(0, min(x.shape[0], self.n * 48), 48)), desc="Creating Plots"):
            prediction = x[sliced : sliced + 48]
            self.heatmap(prediction, classes=bird_classes, file_name=f"{file_list[i]}.png", output_dir=output_dir_heatmaps.as_posix())
            self.top_n_lines(prediction, classes=bird_classes, file_name=f"{file_list[i]}.png", output_dir=output_dir_top_n_lines.as_posix())

        return x

    def heatmap(self, predictions: npt.NDArray[np.floating[Any]], classes: Sequence[str], file_name: str, output_dir: str) -> None:
        """Create a heatmap of the predictions for a 4-minute file.

        :param predictions: The predictions in shape (n_samples=48, n_features=182)
        :param classes: The bird classes
        :param file_name: The name of the file
        :param output_dir: The output directory
        """
        minutes = self.sections_to_time_labels(predictions)

        plt.figure(figsize=(15, 30))
        sns.heatmap(predictions.T, cmap="viridis", yticklabels=classes, xticklabels=5, vmin=0, vmax=1)
        plt.xticks(np.arange(0, 48, 6), labels=minutes[::6], rotation=0)
        plt.title(f"Probability Heatmap of Bird Classes Over Time - {file_name}")
        plt.xlabel("Time Segment (5-second intervals)")
        plt.ylabel("Bird Classes")
        plt.grid(visible=True, which="both", axis="x", linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{file_name}", dpi=300)
        plt.close()

    def top_n_lines(self, predictions: npt.NDArray[np.floating[Any]], classes: Sequence[str], file_name: str, output_dir: str) -> None:
        """Create a line chart of the top N classes over time.

        :param predictions: The predictions in shape (n_samples=48, n_features=182)
        :param classes: The bird classes
        :param file_name: The name of the file
        :param output_dir: The output directory
        """
        threshold = self.threshold
        class_above_threshold = np.max(predictions, axis=0) > threshold
        filtered_predictions = predictions[:, class_above_threshold]
        minutes = self.sections_to_time_labels(predictions)
        plt.figure(figsize=(15, 10))
        for i in range(filtered_predictions.shape[1]):
            plt.plot(range(predictions.shape[0]), filtered_predictions[:, i], label=f"Class {classes[np.where(class_above_threshold)[0][i]]}")
        plt.xticks(np.arange(0, 48, 6), labels=minutes[::6], rotation=0)
        plt.title(f"Top Bird Classes Over Time (Thresholded Probabilities) - {file_name}")
        plt.xlabel("Time Segment (5-second intervals)")
        plt.ylabel("Probability")
        plt.legend(title="Bird Classes", loc="upper right")
        plt.grid(visible=True, which="both", axis="x", linestyle="--", linewidth=0.5)
        plt.savefig(f"{output_dir}/{file_name}", dpi=300)
        plt.close()

    def sections_to_time_labels(self, predictions: npt.NDArray[np.floating[Any]]) -> list[str]:
        """Convert the number of sections to time_labels.

        :param predictions: The predictions in shape (n_samples=48, n_features=182)
        :return: The time labels
        """
        # Time conversion settings
        time_segments = predictions.shape[0]  # Number of time segments
        seconds_per_segment = 5
        total_seconds = time_segments * seconds_per_segment
        return [f"{int(s // 60)} min {int(s % 60)} sec" for s in range(0, total_seconds, seconds_per_segment)]
