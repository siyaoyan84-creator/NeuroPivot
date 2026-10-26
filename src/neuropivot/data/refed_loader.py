from pathlib import Path
import os

import numpy as np
import pandas as pd
from scipy.io import loadmat

def load_subject_data(config, split_df, subject, label_dim):
    cache_dir = config.feature_dir / 'refed_official_1s_preprocessed' / str(subject)
    if not cache_dir.exists():
        alt = config.feature_dir / 'refed_official_1s_preprocessed' / f'{subject:02d}'
        if alt.exists():
            cache_dir = alt
    eeg_path = cache_dir / 'EEG_videos.mat'
    fnirs_path = cache_dir / 'fNIRS_videos.mat'
    if not eeg_path.exists():
        raise FileNotFoundError(eeg_path)
    if not fnirs_path.exists():
        raise FileNotFoundError(fnirs_path)
    subject_rows = split_df[split_df['subject_number'] == subject].drop_duplicates('sample_id').sort_values(['trial_id', 'sample_index']).reset_index(drop=True)
    if len(subject_rows) == 0:
        raise RuntimeError(f'subject {subject} absent from strict sample registry')
    label_col = f'{label_dim}_3c'
    if label_col not in subject_rows.columns:
        raise RuntimeError(f'Missing label column: {label_col}')
    eeg_mat = loadmat(eeg_path)
    fnirs_mat = loadmat(fnirs_path)
    eeg_parts = []
    fnirs_parts = []
    meta_parts = []
    for trial in range(1, 16):
        key = f'video_{trial}'
        if key not in eeg_mat:
            raise RuntimeError(f'Missing EEG key {key}')
        if key not in fnirs_mat:
            raise RuntimeError(f'Missing fNIRS key {key}')
        eeg_trial = np.asarray(eeg_mat[key], dtype=np.float32)
        fnirs_full = np.asarray(fnirs_mat[key], dtype=np.float32)
        if eeg_trial.ndim != 3 or eeg_trial.shape[1:] != (64, 200):
            raise RuntimeError(f'{key}: bad EEG shape {eeg_trial.shape}')
        if fnirs_full.ndim != 4 or fnirs_full.shape[1:] != (6, 51, 25):
            raise RuntimeError(f'{key}: bad fNIRS shape {fnirs_full.shape}')
        fnirs_trial = fnirs_full[:, 2, :, :]
        meta_trial = subject_rows[subject_rows['trial_id'] == trial].sort_values('sample_index').copy()
        n = len(meta_trial)
        if len(eeg_trial) != n:
            raise RuntimeError(f'trial {trial}: EEG={len(eeg_trial)}, registry={n}')
        if len(fnirs_trial) != n:
            raise RuntimeError(f'trial {trial}: fNIRS={len(fnirs_trial)}, registry={n}')
        eeg_parts.append(eeg_trial[:, None, :, :])
        fnirs_parts.append(fnirs_trial[:, None, :, :])
        meta_parts.append(meta_trial)
    eeg = np.concatenate(eeg_parts, axis=0).astype(np.float32, copy=False)
    fnirs = np.concatenate(fnirs_parts, axis=0).astype(np.float32, copy=False)
    metadata = pd.concat(meta_parts, ignore_index=True)
    if len(metadata) != len(eeg):
        raise RuntimeError('metadata / EEG mismatch')
    if len(metadata) != len(fnirs):
        raise RuntimeError('metadata / fNIRS mismatch')
    if not np.isfinite(eeg).all():
        raise RuntimeError('EEG contains NaN/Inf')
    if not np.isfinite(fnirs).all():
        raise RuntimeError('fNIRS contains NaN/Inf')
    y = metadata[label_col].to_numpy(dtype=np.int64)
    if not set(np.unique(y)).issubset({0, 1, 2}):
        raise RuntimeError('Unexpected labels')
    return (eeg, fnirs, y, metadata, eeg_path, fnirs_path)
