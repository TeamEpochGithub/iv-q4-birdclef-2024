
from dataclasses import dataclass
from audiomentations import Compose
import torch

@dataclass
class AudiomentationsCompose:

    compose: Compose = None
    sr: int = 32000

    def __call__(self, x: torch.Tensor):
        augmented_x = x.clone()
        for i in range(x.shape[0]):
            augmented_x[i] = torch.from_numpy(self.compose(x[i].squeeze().numpy(), self.sr))
        return augmented_x