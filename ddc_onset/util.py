from typing import Optional

import numpy as np
import torch

from .spectral import SpectrogramExtractor
from .cnn import SpectrogramNormalizer, PlacementCNN
from .constants import SAMPLE_RATE, Difficulty

_MODULE_SINGLETONS = None


def compute_onset_salience(
    audio: np.ndarray,
    sr: int,
    device: Optional[torch.device] = None,
    difficulty=Difficulty.CHALLENGE,
):
    """Computes onset salience function from audio.

    Args:
        audio: Audio as float32 [num_samples].
        sr: Sample rate of audio. Resamples if not 44.1kHz.
        device: Device to run on. If None, run on CPU.
        difficulty: DDR difficulty label. For general onset detection, the default of Difficulty.CHALLENGE is recommended.
    Returns:
        Onset salience function as float32 [num_frames], frame rate of 100Hz.
    """
    # Check and maybe resample input
    if audio.dtype != np.float32:
        raise TypeError()
    if audio.ndim != 1:
        # TODO: Support multichannel audio
        raise ValueError()
    if sr != SAMPLE_RATE:
        try:
            import resampy
        except ImportError:
            raise Exception(
                "Resampy required to resample audio to 44.1kHz. Please install resampy with `pip install resampy`."
            )
        audio = resampy.resample(audio, sr, SAMPLE_RATE)

    # Load models
    global _MODULE_SINGLETONS
    if _MODULE_SINGLETONS is None:
        _MODULE_SINGLETONS = (
            SpectrogramExtractor(),
            SpectrogramNormalizer(),
            PlacementCNN(),
        )
        _MODULE_SINGLETONS = tuple(model.eval() for model in _MODULE_SINGLETONS)
    if device is not None:
        _MODULE_SINGLETONS = tuple(model.to(device) for model in _MODULE_SINGLETONS)

    # Predict
    with torch.no_grad():
        audio = torch.tensor(audio, device=device).view(1, -1)
        spectrogram = SpectrogramExtractor(audio)
        onset_salience = PlacementCNN(SpectrogramNormalizer(spectrogram))
        return onset_salience.cpu().numpy()


def find_peaks(onset_salience: np.ndarray):
    """Finds peaks in onset salience function."""
    try:
        from scipy.signal import argrelextrema
    except ImportError:
        raise Exception(
            "Scipy required for finding peaks. Please install scipy with `pip install scipy`."
        )
    onset_salience_smoothed = np.convolve(onset_salience, np.hamming(5), "same")
    peaks = argrelextrema(onset_salience_smoothed, np.greater_equal, order=1)[0]
    return peaks


def threshold_peaks(onset_salience: np.ndarray, peaks: np.ndarray, threshold: float):
    """Returns peaks above threshold."""
    return [i for i in peaks if onset_salience[i] >= threshold]
