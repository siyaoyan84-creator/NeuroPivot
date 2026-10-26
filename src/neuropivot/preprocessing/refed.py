from pathlib import Path
import os

"""
REFED official 1-second preprocessing.

Faithful to REFED official 1_process_sample.ipynb.

EEG:
    1000 Hz -> 200 Hz -> 1-second windows

fNIRS:
    0.01-10 Hz IIR
    -> nominal 1000/21 Hz -> 25 Hz
    -> trim final sample iff remainder == 1
    -> 1-second windows
"""
import mne
import numpy as np
EEG_FS = 1000
EEG_FS_DOWN = 200
FNIRS_FS = 1000 / 21
FNIRS_FS_DOWN = 25
FNIRS_L_FREQ = 0.01
FNIRS_H_FREQ = 10.0
WINDOW_SECONDS = 1
EEG_CHANNELS = 64
FNIRS_MEASUREMENTS = 6
FNIRS_CHANNELS = 51
TSMMF_FNIRS_MEASUREMENT_INDEX = 2

class REFEDPreprocessingError(RuntimeError):
    pass

def preprocess_refed_eeg_official(eeg: np.ndarray) -> np.ndarray:
    """
    Input:
        (64, T_raw)

    Output:
        (N, 64, 200)
    """
    eeg = np.asarray(eeg)
    if eeg.ndim != 2:
        raise REFEDPreprocessingError(f'EEG must be 2D, got {eeg.shape}')
    if eeg.shape[0] != EEG_CHANNELS:
        raise REFEDPreprocessingError(f'Expected {EEG_CHANNELS} EEG channels, got {eeg.shape}')
    eeg_down = mne.filter.resample(eeg, up=EEG_FS_DOWN, down=EEG_FS, verbose=False)
    window_samples = WINDOW_SECONDS * EEG_FS_DOWN
    if eeg_down.shape[-1] % window_samples != 0:
        raise REFEDPreprocessingError(f'EEG downsampled length {eeg_down.shape[-1]} is not divisible by {window_samples}')
    eeg_samples = eeg_down.reshape(eeg_down.shape[0], -1, window_samples).transpose(1, 0, 2)
    return eeg_samples

def preprocess_refed_fnirs_official(fnirs: np.ndarray):
    """
    Input:
        (6, 51, T_raw)

    Output:
        samples: (N, 6, 51, 25)
        metadata: dict
    """
    fnirs = np.asarray(fnirs)
    if fnirs.ndim != 3:
        raise REFEDPreprocessingError(f'fNIRS must be 3D, got {fnirs.shape}')
    if fnirs.shape[:2] != (FNIRS_MEASUREMENTS, FNIRS_CHANNELS):
        raise REFEDPreprocessingError(f'Expected fNIRS shape prefix {(FNIRS_MEASUREMENTS, FNIRS_CHANNELS)}, got {fnirs.shape}')
    filtered = mne.filter.filter_data(fnirs, sfreq=FNIRS_FS, l_freq=FNIRS_L_FREQ, h_freq=FNIRS_H_FREQ, method='iir', verbose=False)
    down = mne.filter.resample(filtered, up=FNIRS_FS_DOWN, down=FNIRS_FS, verbose=False)
    before_trim = int(down.shape[-1])
    remainder = before_trim % FNIRS_FS_DOWN
    trim_samples = 0
    if remainder != 0:
        if remainder == 1:
            down = down[..., :-1]
            trim_samples = 1
        else:
            raise REFEDPreprocessingError(f'fNIRS length cannot be divided into 1-second windows after resampling: length={before_trim}, remainder={remainder}')
    window_samples = WINDOW_SECONDS * FNIRS_FS_DOWN
    samples = down.reshape(down.shape[0], down.shape[1], -1, window_samples).transpose(2, 0, 1, 3)
    metadata = {'raw_samples': int(fnirs.shape[-1]), 'resampled_samples_before_trim': before_trim, 'trim_samples': trim_samples, 'resampled_samples_after_trim': int(down.shape[-1]), 'n_windows': int(samples.shape[0])}
    return (samples, metadata)

def refed_tsmmf_inputs(eeg_samples: np.ndarray, fnirs_samples: np.ndarray):
    """
    Match the official supervised loader +
    process_data() shapes.

    EEG:
        (N,64,200)
        -> (N,1,64,200)

    fNIRS:
        select measurement index 2:
        (N,6,51,25)
        -> (N,51,25)
        -> (N,1,51,25)
    """
    if len(eeg_samples) != len(fnirs_samples):
        raise REFEDPreprocessingError('EEG/fNIRS sample count mismatch')
    eeg = np.expand_dims(eeg_samples, axis=1)
    fnirs = fnirs_samples[:, TSMMF_FNIRS_MEASUREMENT_INDEX, :, :]
    fnirs = np.expand_dims(fnirs, axis=1)
    return (eeg.astype(np.float32), fnirs.astype(np.float32))
