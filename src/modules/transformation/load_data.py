"""Example transformation block for the transformation pipeline."""
from dataclasses import dataclass

import dask
import dask.array as da
import librosa
import numpy as np

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock
from src.typing.typing import XData


@dataclass
class LoadXData(VerboseTransformationBlock):
    """An example transformation block for the transformation pipeline."""

    bird_2024: str

    def custom_transform(self, data: XData) -> XData:
        """Apply a custom transformation to the data.

        :param data: The data to transform
        :param kwargs: Any additional arguments
        :return: The transformed data
        """
        filenames_2024 = data.meta_2024.filename
        # Add path before the list of filenames
        filenames_2024 = [self.bird_2024 + filename for filename in filenames_2024]
        keys_2024 = data.meta_2024.samplename
        test = self.create_dask_array_from_files(filenames_2024)

        test = {key: self.set_nan_to_zero(value) for key, value in test.items()}

        test = {key: self.take_first_5_seconds(value) for key, value in test.items()}

        test = self.get_first_n_items(test)
        # Compute test
        test = dask.compute(test)

        # Set the bird_2024 attribute of the data
        data.bird_2024 = test
        return data

    import dask


    @dask.delayed
    def load_lazy_audio(self, path: str) -> da.Array:
        """Load audio data lazily using librosa"""

        audio = librosa.load(path, sr=32000, dtype=np.float32)[0]
        return audio


    def create_dask_array_from_files(self, file_paths, dtype='float32'):
        # Read all files as delayed
        delayed_audio_data = [self.load_lazy_audio(fp) for fp in file_paths]
        # Create a Dask bag from the delayed objects
        audio_bag = dask.bag.from_delayed(delayed_audio_data)
        # If needed, convert to Dask array (consider how you want to handle differing lengths)
        # This example assumes you will process them as they are or have a function to normalize lengths
        # Note: Converting directly to an array might require uniform chunks or handling via padding/masking
        # audio_array = concatenate([da.from_array(x, chunks=(100000,)) for x in audio_bag])
        return audio_bag

    # Usage example
    file_paths = ['audio1.ogg', 'audio2.ogg', ...]  # Replace with your actual file paths
    dask_audio_bag = create_dask_array_from_files(file_paths)


        # def load_lazy_audio(self, path: str) -> da.Array:

    #     """Load audio data lazily using librosa"""
    #
    #     # Check if audio exists as numpy, else load via librosa and save
    #     try:
    #         # Load numpy array with dask
    #         audio = da.from_npy_stack(path.replace('.ogg', '.npy'))
    #         logger.info("Loaded audio from numpy array")
    #
    #     except:
    #         audio = librosa.load(path, sr=32000, dtype=np.float32)[0]
    #
    #         # Save audio as numpy array at the path, but extension should be .npy
    #         np.save(path.replace('.ogg', '.npy'), audio)
    #
    #     # Convert to dask array
    #     audio = da.from_array(audio, chunks=32000)
    #
    #     return audio

    @dask.delayed
    def set_nan_to_zero(self, data: list) -> list:
        """Set nan values to zero"""
        return [np.nan_to_num(x) for x in data]

    def take_first_5_seconds(self, data: np.ndarray) -> np.ndarray:
        """Take the first 5 seconds of the data"""
        return data[:5 * 32000]

    def get_first_n_items(self, d, n=1):
        from itertools import islice
        return dict(islice(d.items(), n))
