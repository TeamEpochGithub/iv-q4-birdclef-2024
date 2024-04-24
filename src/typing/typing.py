"""Common type definitions for the project."""
from dataclasses import dataclass
from typing import Any

import numpy.typing as npt
import pandas as pd
import pandas as pd
import numpy as np


@dataclass
class XData:
    """Dataclass to hold X data.

    :param meta_2024: Metadata of BirdClef2024:
    :param meta_2023: Metadata of BirdClef2023:
    :param meta_2022: Metadata of BirdClef2022:
    :param meta_2021: Metadata of BirdClef2021:
    :param bird_2024: Audiodata of BirdClef2024
    :param bird_2023: Audiodata of BirdClef2023
    :param bird_2022: Audiodata of BirdClef2022
    :param bird_2021: Audiodata of BirdClef2021
    """

    meta_2024: pd.DataFrame
    meta_2023: pd.DataFrame | None = None
    meta_2022: pd.DataFrame | None = None
    meta_2021: pd.DataFrame | None = None
    bird_2024: npt.NDArray[Any] | None = None
    bird_2023: npt.NDArray[Any] | None = None
    bird_2022: npt.NDArray[Any] | None = None
    bird_2021: npt.NDArray[Any] | None = None

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        return "XData"

    def __getitem__(self, indexer) -> "XData":
        if isinstance(indexer, dict):
            sliced_fileds = {}
            # Slice all the years by the appropriate indices and save to a dict
            for year in indexer:
                if getattr(self,f"bird_{year}") is not None:
                    sliced_fileds[f"bird_{year}"] = getattr(self,f"bird_{year}")[indexer[year]]
                if getattr(self, f"meta_{year}") is not None:
                    sliced_fileds[f"meta_{year}"] = getattr(self,f"meta_{year}")[indexer[year]]
            return XData(**sliced_fileds)
        elif isinstance(indexer, str):
            # allow dict like indexing with keys
            return getattr(self, indexer)

        else:
            # Needed for the main trainer to instantiate datasets properly
            sliced_meta_2024 = self.meta_2024.iloc[indexer]
            sliced_bird_2024 = self.bird_2024[indexer]

            return XData(
                meta_2024=sliced_meta_2024,
                bird_2024=sliced_bird_2024
            )



@dataclass
class YData:
    """Dataclass to hold X data.

    :param meta_2024: Metadata of BirdClef2024:
    :param meta_2023: Metadata of BirdClef2023:
    :param meta_2022: Metadata of BirdClef2022:
    :param meta_2021: Metadata of BirdClef2021:
    :param label_2024: Labels of BirdClef2024
    :param label_2023: Labels of BirdClef2023
    :param label_2022: Labels of BirdClef2022
    :param label_2021: Labels of BirdClef2021
    """

    meta_2024: pd.DataFrame
    meta_2023: pd.DataFrame | None = None
    meta_2022: pd.DataFrame | None = None
    meta_2021: pd.DataFrame | None = None
    label_2024: pd.DataFrame | None = None
    label_2023: pd.DataFrame | None = None
    label_2022: pd.DataFrame | None = None
    label_2021: pd.DataFrame | None = None

    def __getitem__(self, indexer):
        if isinstance(indexer, dict):
            sliced_fileds = {}
            # Slice all the years by the appropriate indices and save to a dict
            for year in indexer:
                if getattr(self,f"label_{year}") is not None:
                    sliced_fileds[f"label_{year}"] = getattr(self,f"label_{year}").iloc[indexer[year]]
                if getattr(self, f"meta_{year}") is not None:
                    sliced_fileds[f"meta_{year}"] = getattr(self,f"meta_{year}").iloc[indexer[year]]
            return YData(**sliced_fileds)
        
        elif isinstance(indexer, str):
            # allow dict like indexing with keys
            return getattr(self, indexer)

        else:
            # If indices are not a dict assume that we are using the 2024 data 
            sliced_meta_2024 = self.meta_2024.iloc[indexer]
            sliced_label_2024 = self.label_2024.iloc[indexer]

            return YData(
                meta_2024=sliced_meta_2024,
                label_2024=sliced_label_2024
            )


    def __repr__(self) -> str:
        """Return a string representation of the object."""
        return "YData"

# if __name__ == "__main__":
#     # Create dummy data for metadata
#     meta_2024_data = {
#         'id': [1, 2, 3],
#         'name': ['Bird1', 'Bird2', 'Bird3'],
#         'species': ['Species1', 'Species2', 'Species3']
#     }
#     meta_2024 = pd.DataFrame(meta_2024_data)

#     # Create dummy data for audiodata
#     bird_2024_data = np.random.rand(3, 10)  # Assuming 3 samples with 10 features each
#     bird_2024 = bird_2024_data.astype(np.float32)

#     # Instantiate XData object with the dummy data
#     X = XData(
#         meta_2024=meta_2024,
#         bird_2024=bird_2024
#     )
#     print(X)

#         # Create dummy data for metadata
#     meta_2024_data = {
#         'id': [1, 2, 3],
#         'name': ['Bird1', 'Bird2', 'Bird3'],
#         'species': ['Species1', 'Species2', 'Species3']
#     }
#     meta_2024 = pd.DataFrame(meta_2024_data)

#     # Create dummy data for labels
#     label_2024_data = np.random.randint(0, 2, size=(3,182))  # Assuming 3 samples with binary labels
#     label_2024 = label_2024_data.astype(np.int32)

#     # Instantiate YData object with the dummy data
#     Y = YData(
#         meta_2024=meta_2024,
#         label_2024=label_2024
#     )
#     print(Y)