"""When using the re-scraped metadata, some primary labels have been fixed.
This script is used to reflect those fixes in the audio file names in the train_audio directory."""
from pathlib import Path
from shutil import copy

FIXES = {
    "marsan/XC184468.ogg": "woosan/XC184468.ogg",
    "commoo3/XC724832.ogg": "eurcoo/XC724832.ogg",
    "comkin1/XC460945.ogg": "eurcoo/XC460945.ogg",
    "purher1/XC445303.ogg": "graher1/XC445303.ogg",
    "comsan/XC589069.ogg": "grnsan/XC589069.ogg",
    "woosan/XC371015.ogg": "bkwsti/XC371015.ogg",
}


TRAIN_AUDIO_PATH = Path("./data/raw/train_audio/")

def fix_audio_primary_labels() -> None:
    """Copy the audio files with fixed primary labels to the correct directory."""
    for audio_path in FIXES.keys():
        full_audio_path = TRAIN_AUDIO_PATH / audio_path
        new_path = TRAIN_AUDIO_PATH / FIXES[str(audio_path)]
        print(f"Copying {full_audio_path} to {new_path}")
        copy(full_audio_path.as_posix(), new_path.as_posix())

if __name__ == "__main__":
    fix_audio_primary_labels()
