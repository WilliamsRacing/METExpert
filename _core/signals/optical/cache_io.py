"""Read immutable whole or sliced numerical spectra through one contract."""
from pathlib import Path
import numpy as np


def load_spectrogram(record):
    path = Path(record["_spectrogram_path"])
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    selection = record.get("spectrogram_slice")
    if selection is not None:
        if (not isinstance(selection, (list, tuple)) or len(selection) != 2 or
                any(isinstance(i, bool) or not isinstance(i, int) for i in selection)):
            raise ValueError("Malformed spectrogram_slice")
        start, stop = selection
        if not 0 <= start < stop <= len(values):
            raise ValueError("Spectrogram slice is out of bounds")
        values = values[start:stop]
    if values.shape != (int(record["n_frames"]), 3, 37, 23):
        raise ValueError("Spectrum shape/slice does not match record frames: " + str(path))
    return values
