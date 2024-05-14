"""Wrapper class to create spectrograms from the data."""

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torchaudio.transforms import MelSpectrogram
from torchaudio.functional import create_dct


@dataclass
class Spec:
    """Wrapper class for spectrogram functions from torchaudio."""

    spec: functools.partial[torch.nn.Module]
    output_shape: tuple[int, int] = (224, 224)  # H, W
    seqeunce_length: int = 160000  # 5 seconds x 32kHz
    scale: Callable[[torch.Tensor], torch.Tensor] = torch.log10
    sample_rate: int = 32000

    def __post_init__(self) -> None:
        """Calculate the params for the desired input shape and instantiate the spec class."""
        self.n_fft = self.output_shape[0] * 2 - 1
        self.hop_length = self.seqeunce_length // self.output_shape[1] + 1
        # Re-instantiate spec class with params for deisred output shape
        if self.spec.func is MelSpectrogram:
            self.instantiated_spec = self.spec(
                n_fft=self.n_fft * 4,
                hop_length=self.hop_length,
                n_mels=self.output_shape[0],
                sample_rate=self.sample_rate,
            )
        else:
            self.instantiated_spec = self.spec(n_fft=self.n_fft, hop_length=self.hop_length)


    def __call__(self, input_data: torch.Tensor) -> torch.Tensor:
        """Create spectrograms from the input."""
        # Create spectrograms from the input

        spec_out = self.instantiated_spec(input_data)

        if len(spec_out.shape) == 3:
            spec_out = spec_out.unsqueeze(1)
        
        dct_matrix = create_dct(self.output_shape[0], self.output_shape[0], norm=None)
        mfcc = torch.matmul(dct_matrix, spec_out)
        # Log spec
        spec_out = torch.nan_to_num(torch.log10(spec_out), neginf=10e-10, posinf=1, nan=0)

        # log10 returns some -inf's replace them with low amplitude
        return torch.cat([spec_out, mfcc], dim=1)
