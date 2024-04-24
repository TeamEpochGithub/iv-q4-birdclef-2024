"""Wrapper class to create spectrograms from the data."""

from collections.abc import Callable
from dataclasses import dataclass
import functools
import torch


@dataclass
class Spec:
    spec: functools.partial[torch.nn.Module]
    output_shape: tuple[int,int] = (224,224) # H, W
    seqeunce_length: int = 160000 # 5 seconds x 32kHz
    scale: Callable = torch.log

    def __post_init__(self):
        n_fft = self.output_shape[0]*2-1
        hop_length = self.seqeunce_length // self.output_shape[1] + 1
        # Re-instantiate spec class with params for deisred output shape
        self.instantiated_spec = self.spec(n_fft=n_fft, hop_length=hop_length)
            
    def __call__(self, input_data: torch.Tensor):
        # Create spectrograms from the input
        spec_out = self.instantiated_spec(input_data).unsqueeze(1)

        return self.scale(spec_out)