"""Abstract scorer class from which other scorers inherit from."""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Iterable

import numpy.typing as npt
from typing_extensions import override

from src.scoring.scorer import Scorer
from src.typing.typing import YData


@dataclass
class Accuracy(Scorer):
    """Abstract scorer class from which other scorers inherit from.

    :param name: The name of the scorer.
    """

    name: str

    def __init__(self, name: str) -> None:
        """Initialize the scorer with a name.

        :param name: The name of the scorer.
        """
        self.name = name

    @override
    def __call__(self, y_true: YData, y_pred: npt.NDArray[Any], **kwargs: Any) -> float:
        """Calculate the score.

        :param y_true: The true labels.
        :param y_pred: The predicted labels.
        :param kwargs: Additional keyword arguments.
        :return: The calculated score.
        """
        # Calculate the accuracy score
        # Apply a threshold to the predictions
        test_indices: Mapping[str, Sequence[int]] = kwargs["test_indices"]
        years: Iterable[str] = kwargs["years"]
        output_dir: str = kwargs.get("output_dir", "")
        y_true = y_true[f"label_{years[0]}"].iloc[test_indices[str(years[0])]]

        y_pred = (y_pred > 0.4125).astype(int)

        # Save the confusion matrix and save to output dir as a png using seaborn
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        import seaborn as sns
        import matplotlib.pyplot as plt

        # Ensure no scientific notation in annot
        sns.heatmap(cm, annot=True, fmt='d')

        # Set x and y axis of heatmap
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(output_dir.as_posix() + '/confusion_matrix.png')

        # Calculate f1 score
        from sklearn.metrics import f1_score
        f1 = f1_score(y_true, y_pred)
        return f1

    def __str__(self) -> str:
        """Return the name of the scorer.

        :return: The name of the scorer.
        """
        return self.name
