from dataclasses import dataclass
from typing import Any

import torch
import audiomentations

@dataclass
class AddBackgroundNoiseWrapper():
    p: float = 0.5
    sounds_path: str = 'data/raw/esc50/audio/'
    min_snr_db: float = -3.0
    max_snr_db: float = 3.0
    noise_transform: Any = audiomentations.PolarityInversion(p=0.5)
    aug: Any = None

    def __post_init__(self):
        if torch.cuda.is_available():
            self.aug = audiomentations.AddBackgroundNoise(p=self.p, sounds_path=self.sounds_path,
                                                          min_snr_db=self.max_snr_db,
                                                          max_snr_db=self.max_snr_db, 
                                                          noise_transform=self.noise_transform)
        else:
            self.aug = None
        self.__dict__ = {'Placeholder': 'Remove Later'}

    def __call__(self, x, sr):
        if self.aug is not None:
            return self.aug(x, sr)
        else:
            return x
        
    
