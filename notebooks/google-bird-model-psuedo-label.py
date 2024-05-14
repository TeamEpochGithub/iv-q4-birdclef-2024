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
from typing import Any

AUDIO_PATH = Path("data/raw/2024/test_soundscapes")
FILE_PATHS = list(AUDIO_PATH.glob("*.ogg"))
MODEL_PATH = "https://kaggle.com/models/google/bird-vocalization-classifier/frameworks/TensorFlow2/variations/bird-vocalization-classifier/versions/4"
PREDICTIONS_OUTPUT_PATH = Path("data/raw/2024/google-bvc-predictions.csv")
CLASSES = ["asbfly", "ashdro1", "ashpri1", "ashwoo2", "asikoe2", "asiope1", "aspfly1", "aspswi1", "barfly1", "barswa", "bcnher", "bkcbul1", "bkrfla1", "bkskit1", "bkwsti", "bladro1", "blaeag1", "blakit1", "blhori1", "blnmon1", "blrwar1", "bncwoo3", "brakit1", "brasta1", "brcful1", "brfowl1", "brnhao1", "brnshr", "brodro1", "brwjac1", "brwowl1", "btbeat1", "bwfshr1", "categr", "chbeat1", "cohcuc1", "comfla1", "comgre", "comior1", "comkin1", "commoo3", "commyn", "compea", "comros", "comsan", "comtai1", "copbar1", "crbsun2", "cregos1", "crfbar1", "crseag1", "dafbab1", "darter2", "eaywag1", "emedov2", "eucdov", "eurbla2", "eurcoo", "forwag1", "gargan", "gloibi", "goflea1", "graher1", "grbeat1", "grecou1", "greegr", "grefla1", "grehor1", "grejun2", "grenig1", "grewar3", "grnsan", "grnwar1", "grtdro1", "gryfra", "grynig2", "grywag", "gybpri1", "gyhcaf1", "heswoo1", "hoopoe", "houcro1", "houspa", "inbrob1", "indpit1", "indrob1", "indrol2", "indtit1", "ingori1", "inpher1", "insbab1", "insowl1", "integr", "isbduc1", "jerbus2", "junbab2", "junmyn1", "junowl1", "kenplo1", "kerlau2", "labcro1", "laudov1", "lblwar1", "lesyel1", "lewduc1", "lirplo", "litegr", "litgre1", "litspi1", "litswi1", "lobsun2", "maghor2", "malpar1", "maltro1", "malwoo1", "marsan", "mawthr1", "moipig1", "nilfly2", "niwpig1", "nutman", "orihob2", "oripip1", "pabflo1", "paisto1", "piebus1", "piekin1", "placuc3", "plaflo1", "plapri1", "plhpar1", "pomgrp2", "purher1", "pursun3", "pursun4", "purswa3", "putbab1", "redspu1", "rerswa1", "revbul", "rewbul", "rewlap1", "rocpig", "rorpar", "rossta2", "rufbab3", "ruftre2", "rufwoo2", "rutfly6", "sbeowl1", "scamin3", "shikra1", "smamin1", "sohmyn1", "spepic1", "spodov", "spoowl1", "sqtbul1", "stbkin1", "sttwoo1", "thbwar1", "tibfly3", "tilwar1", "vefnut1", "vehpar1", "wbbfly1", "wemhar1", "whbbul2", "whbsho3", "whbtre1", "whbwag1", "whbwat1", "whbwoo2", "whcbar1", "whiter2", "whrmun", "whtkin2", "woosan", "wynlau1", "yebbab1", "yebbul3", "zitcis1"]

SAMPLE_RATE = 32000
WINDOW = 5 * SAMPLE_RATE

# %%

model = hub.load(MODEL_PATH)
model_labels_df = pd.read_csv(hub.resolve(MODEL_PATH) + "/assets/label.csv")

# %%

label_to_index = {v: k for k, v in enumerate(CLASSES)}
model_labels = {v: k for k, v in enumerate(model_labels_df.ebird2021)}
model_bc_indexes = [model_labels[label] if label in model_labels else -1 for label in CLASSES]

# Filter out birds that the model doesn't predict
missing_birds = set(np.array(CLASSES)[np.array(model_bc_indexes) == -1])
missing_birds

# %%

class AudioDataset(Dataset):
    def __len__(self):
        return len(FILE_PATHS)
    def __getitem__(self, i):
        filename = FILE_PATHS[i].name
        audio = torchaudio.load(FILE_PATHS[i])[0].numpy()[0]
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

# %%
from scipy.special import expit

logits = np.reshape(np.stack(list(all_predictions.values())), (-1, len(CLASSES)))
submission = pd.DataFrame(expit(logits), columns=CLASSES)
submission.to_csv(PREDICTIONS_OUTPUT_PATH, index=False)
