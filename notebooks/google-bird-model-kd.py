import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Any, Final

METADATA_PATH: Final[Path] = Path("data/raw/train_metadata.csv")
AUDIO_PATH: Final[Path] = Path("data/raw/2024/train_audio")
MODEL_PATH: str = "https://kaggle.com/models/google/bird-vocalization-classifier/frameworks/TensorFlow2/variations/bird-vocalization-classifier/versions/4"
PREDICTIONS_OUTPUT_PATH: Final[Path] = Path("data/raw/2024/kd/google/bird-vocalization-classifier-submission.csv")

SAMPLE_RATE: Final[int] = 32000
WINDOW: Final[int] = 5 * SAMPLE_RATE

# %%

metadata = pd.read_csv(METADATA_PATH)
model = hub.load(MODEL_PATH)
model_labels_df = pd.read_csv(hub.resolve(MODEL_PATH) + "/assets/label.csv")

# %%

# Test if using GPUs is possible
# gpus = tf.config.experimental.list_physical_devices("GPU")
# print("Num GPUs Available: ", len(gpus))
# for gpu in gpus:
#     print("Name:", gpu.name, "Type:", gpu.device_type)

# %%

# if gpus:
#     try:
#         # Simple computation to test GPU utilization
#         with tf.device("/GPU:0"):  # Change this if you have more than one GPU
#             a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
#             b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
#             c = tf.matmul(a, b)
#             print(c)
#     except RuntimeError as e:
#         print(e)
# else:
#     print("No GPU available")


# %%

index_to_label = sorted(metadata.primary_label.unique())
label_to_index = {v: k for k, v in enumerate(index_to_label)}
model_labels = {v: k for k, v in enumerate(model_labels_df.ebird2021)}
model_bc_indexes = [model_labels[label] if label in model_labels else -1 for label in index_to_label]

# Filter out birds that the model doesn't predict
missing_birds = set(np.array(index_to_label)[np.array(model_bc_indexes) == -1])
missing_birds

# %%

# Use a torch dataloader to decode audio in parallel on CPU while GPU is running
class AudioDataset(Dataset):
    def __len__(self):
        return len(metadata)
    def __getitem__(self, i):
        filename = metadata.filename[i]
        audio = torchaudio.load(AUDIO_PATH / filename)[0].numpy()[0]
        return audio, filename
dataloader = DataLoader(AudioDataset(), batch_size=1, num_workers=os.cpu_count())

all_predictions: dict[str, npt.NDArray[np.floating[Any]]] = {}

with tf.device("/GPU:0"):
    for audio, filename in tqdm(dataloader):
        audio = audio[0]
        filename = filename[0]
        file_predictions: list[npt.NDArray[np.floating[Any]]] = []
        for i in range(0, len(audio), WINDOW):
            clip = audio[i:i+WINDOW]
            if len(clip) < WINDOW:
                clip = np.concatenate([clip, np.zeros(WINDOW - len(clip))])
            result = model.infer_tf(clip[None, :])
            prediction = np.concatenate([result[0].numpy(), -100], axis=None)  # add -100 logit for unpredicted birds
            file_predictions.append(prediction[model_bc_indexes])
        all_predictions[filename] = np.stack(file_predictions)

torch.save(all_predictions, "./tm/predictions.pt")

# %%

import sklearn.metrics

"""
This script exists to reduce code duplication across metrics.
"""

import numpy as np
import pandas as pd
import pandas.api.types


class ParticipantVisibleError(Exception):
    pass


class HostVisibleError(Exception):
    pass


def treat_as_participant_error(error_message: str, solution: pd.DataFrame | np.ndarray) -> bool:
    """Many metrics can raise more errors than can be handled manually. This function attempts
    to identify errors that can be treated as ParticipantVisibleError without leaking any competition data.

    If the solution is purely numeric, and there are no numbers in the error message,
    then the error message is sufficiently unlikely to leak usable data and can be shown to participants.

    We expect this filter to reject many safe messages. It's intended only to reduce the number of errors we need to manage manually.
    """
    # This check treats bools as numeric
    if isinstance(solution, pd.DataFrame):
        solution_is_all_numeric = all([pandas.api.types.is_numeric_dtype(x) for x in solution.dtypes.values])
        solution_has_bools = any([pandas.api.types.is_bool_dtype(x) for x in solution.dtypes.values])
    elif isinstance(solution, np.ndarray):
        solution_is_all_numeric = pandas.api.types.is_numeric_dtype(solution)
        solution_has_bools = pandas.api.types.is_bool_dtype(solution)

    if not solution_is_all_numeric:
        return False

    for char in error_message:
        if char.isnumeric():
            return False
    if solution_has_bools:
        if "true" in error_message.lower() or "false" in error_message.lower():
            return False
    return True


def safe_call_score(metric_function, solution, submission, **metric_func_kwargs):
    """Call score. If that raises an error and that already been specifically handled, just raise it.

    Otherwise, make a conservative attempt to identify potential participant visible errors.
    """
    try:
        score_result = metric_function(solution, submission, **metric_func_kwargs)
    except Exception as err:
        error_message = str(err)
        if err.__class__.__name__ == "ParticipantVisibleError":
            raise ParticipantVisibleError(error_message)
        elif err.__class__.__name__ == "HostVisibleError":
            raise HostVisibleError(error_message)
        else:
            if treat_as_participant_error(error_message, solution):
                raise ParticipantVisibleError(error_message)
            else:
                raise err
    return score_result



def roc_auc(solution: pd.DataFrame, submission: pd.DataFrame) -> float: #row_id_column_name: str
    """Version of macro-averaged ROC-AUC score that ignores all classes that have no true positive labels.
    """
    #del solution[row_id_column_name]
    #del submission[row_id_column_name]

    if not pandas.api.types.is_numeric_dtype(submission.values):
        bad_dtypes = {x: submission[x].dtype  for x in submission.columns if not pandas.api.types.is_numeric_dtype(submission[x])}
        raise ParticipantVisibleError(f"Invalid submission data types found: {bad_dtypes}")

    solution_sums = solution.sum(axis=0)
    scored_columns = list(solution_sums[solution_sums > 0].index.values)
    assert len(scored_columns) > 0

    return safe_call_score(sklearn.metrics.roc_auc_score, solution[scored_columns].values, submission[scored_columns].values, average='macro')


# test the score function
x = pd.get_dummies(metadata.primary_label)
y = x.copy()
y[index_to_label] = 0
roc_auc(x, x), roc_auc(x, y)

# %%

actual_classes = torch.tensor([label_to_index[label] for label in metadata.primary_label])
logits = torch.stack([torch.tensor(row[0]) for row in all_predictions.values()])
actual_probs = torch.eye(len(index_to_label))[actual_classes]
solution = pd.DataFrame(actual_probs.numpy(), columns=index_to_label)
submission = pd.DataFrame(torch.sigmoid(logits).numpy(), columns=index_to_label)
submission.to_csv(PREDICTIONS_OUTPUT_PATH, index=False)

roc_auc(
    solution=solution,
    submission=pd.DataFrame(torch.sigmoid(logits).numpy(), columns=index_to_label),
)
