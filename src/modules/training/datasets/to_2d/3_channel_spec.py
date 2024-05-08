"""Wrapper class to create spectrograms from the data."""

import functools
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torchaudio.transforms import MelSpectrogram


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
            self.instantiated_spec1 = self.spec(
                n_fft=self.n_fft * 6,
                hop_length=self.hop_length,
                n_mels=self.output_shape[0],
                sample_rate=self.sample_rate,
                f_min=0,
                f_max=6000,
            )
            self.instantiated_spec2 = self.spec(
                n_fft=self.n_fft * 6,
                hop_length=self.hop_length,
                n_mels=self.output_shape[0],
                sample_rate=self.sample_rate,
                f_min=6000,
                f_max=11000,
            )
            self.instantiated_spec3 = self.spec(
                n_fft=self.n_fft * 6,
                hop_length=self.hop_length,
                n_mels=self.output_shape[0],
                sample_rate=self.sample_rate,
                f_min=11000,
                f_max=16000,
            )
        else:
            self.instantiated_spec = self.spec(n_fft=self.n_fft, hop_length=self.hop_length)

    def __call__(self, input_data: torch.Tensor) -> torch.Tensor:
        """Create spectrograms from the input."""
        # Create spectrograms from the input

        spec_out1 = self.instantiated_spec1(input_data)
        spec_out2 = self.instantiated_spec2(input_data)
        spec_out3 = self.instantiated_spec3(input_data)
        if len(spec_out1.shape) == 3:
            spec_out1 = spec_out1.unsqueeze(1)
            spec_out2 = spec_out2.unsqueeze(1)
            spec_out3 = spec_out3.unsqueeze(1)

        spec_out = torch.cat((spec_out1, spec_out2, spec_out3), dim=1)

        # log10 returns some -inf's replace them with zeros
        return torch.nan_to_num(self.scale(spec_out + 10**-10), neginf=-10, posinf=1, nan=0)
