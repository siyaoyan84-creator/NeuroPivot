from pathlib import Path
import os
'EAV strict per-trial preprocessing for the retained main method.'
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import scipy.io
from scipy import signal
EAV_RAW_FS = 500
EAV_TARGET_FS = 100
EAV_RAW_SAMPLES = 10000
EAV_TARGET_SAMPLES = 2000
EAV_CHANNELS = 30
EAV_TRIALS_PER_SUBJECT = 200
EAV_WINDOW_SECONDS = 5
EAV_WINDOW_SAMPLES = 500
EAV_WINDOWS_PER_TRIAL = 4
EAV_EMOTION_ORDER = ['neutral', 'sadness', 'anger', 'happiness', 'calmness']
EAV_PREPROCESSING_PROTOCOLS = {'eav_strict_pertrial_preproc_v1'}

class EAVPreprocessingError(Exception):
    pass

def _validate_raw(raw: np.ndarray) -> None:
    expected = (EAV_RAW_SAMPLES, EAV_CHANNELS, EAV_TRIALS_PER_SUBJECT)
    if raw.shape != expected:
        raise EAVPreprocessingError(f'Expected raw EAV EEG shape {expected}, got {raw.shape}')

def _bandpass_sos():
    return signal.butter(5, [3, 50], btype='bandpass', fs=EAV_RAW_FS, output='sos')

def load_eav_eeg(eeg_path: str | Path, eeg_variable: str) -> np.ndarray:
    path = Path(eeg_path)
    if not path.exists():
        raise FileNotFoundError(path)
    mat = scipy.io.loadmat(path)
    if eeg_variable not in mat:
        raise EAVPreprocessingError(f'{path}: variable {eeg_variable!r} not found')
    raw = np.asarray(mat[eeg_variable])
    _validate_raw(raw)
    return raw

def preprocess_strict_per_trial(raw: np.ndarray) -> np.ndarray:
    """
    Leakage-safe independent parent-trial preprocessing.

    Returns
    -------
    np.ndarray
        Shape: (200, 2000, 30)
    """
    _validate_raw(raw)
    sos = _bandpass_sos()
    output = np.empty((EAV_TRIALS_PER_SUBJECT, EAV_TARGET_SAMPLES, EAV_CHANNELS), dtype=np.float64)
    for trial_index in range(EAV_TRIALS_PER_SUBJECT):
        trial = raw[:, :, trial_index]
        filtered = signal.sosfilt(sos, trial, axis=0)
        downsampled = signal.resample_poly(filtered, up=1, down=5, axis=0)
        if downsampled.shape != (EAV_TARGET_SAMPLES, EAV_CHANNELS):
            raise EAVPreprocessingError(f'Trial {trial_index}: unexpected shape {downsampled.shape}')
        output[trial_index] = downsampled
    return output

def preprocess_eav_subject(raw: np.ndarray, protocol: str) -> np.ndarray:
    if protocol != 'eav_strict_pertrial_preproc_v1':
        raise EAVPreprocessingError('Only the main strict per-trial protocol is supported')
    return preprocess_strict_per_trial(raw)

def segment_eav_trials(trials: np.ndarray) -> np.ndarray:
    """
    Convert preprocessed parent trials into EEGNet windows.

    Input:
        (N, 2000, 30)

    Output:
        (N*4, 30, 500, 1)
    """
    if trials.ndim != 3:
        raise EAVPreprocessingError(f'Expected 3D trials, got {trials.shape}')
    if trials.shape[1:] != (EAV_TARGET_SAMPLES, EAV_CHANNELS):
        raise EAVPreprocessingError(f'Expected (*, 2000, 30), got {trials.shape}')
    n_trials = trials.shape[0]
    windows = trials.reshape(n_trials, EAV_WINDOWS_PER_TRIAL, EAV_WINDOW_SAMPLES, EAV_CHANNELS)
    windows = windows.transpose(0, 1, 3, 2)
    return windows.reshape(n_trials * EAV_WINDOWS_PER_TRIAL, EAV_CHANNELS, EAV_WINDOW_SAMPLES, 1)
