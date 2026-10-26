from pathlib import Path
import os

"""
REFED 1-second sample identity and official sample-level split logic.

No signal arrays are stored here.

Frozen preprocessing protocol:
    refed_official_1s_labelanchored_v1
"""
from typing import Dict, Any
import numpy as np
import pandas as pd
import scipy.io
PREPROCESSING_PROTOCOL = 'refed_official_1s_labelanchored_v1'
EEG_FS_RAW = 1000
EEG_FS_DOWN = 200
FNIRS_FS_DOWN = 25
EEG_CHANNELS = 64
FNIRS_MEASUREMENTS = 6
FNIRS_CHANNELS = 51
FNIRS_TSMMF_MEASUREMENT_INDEX = 2

class REFEDSampleError(RuntimeError):
    pass

def _subject_number(value) -> int:
    text = str(value)
    if text.startswith('subject'):
        text = text.replace('subject', '', 1)
    return int(float(text))

def label_to_3c_official(values: np.ndarray) -> np.ndarray:
    """
    Match REFED official label_to_3c():

        low    if label < 256 * 0.3
        medium otherwise
        high   if label > 256 * 0.7
    """
    x = np.asarray(values)
    out = np.ones(x.shape, dtype=np.int64)
    out[x < 256 * 0.3] = 0
    out[x > 256 * 0.7] = 2
    return out

def build_refed_sample_registry(manifest: pd.DataFrame, rounding_audit: pd.DataFrame) -> pd.DataFrame:
    required_manifest = {'subject_id', 'trial_id', 'trial_group_id', 'fnirs_samples', 'label_path', 'label_variable', 'label_steps'}
    missing = required_manifest - set(manifest.columns)
    if missing:
        raise REFEDSampleError(f'Manifest missing columns: {sorted(missing)}')
    required_rounding = {'subject_id', 'trial_id', 'fnirs_raw_samples', 'resampled_samples', 'label_anchored_target', 'delta_samples'}
    missing = required_rounding - set(rounding_audit.columns)
    if missing:
        raise REFEDSampleError(f'Rounding audit missing columns: {sorted(missing)}')
    manifest = manifest.copy()
    manifest['subject_number_norm'] = manifest['subject_id'].map(_subject_number)
    rounding = rounding_audit.copy()
    rounding['subject_number_norm'] = rounding['subject_id'].map(_subject_number)
    rounding_lookup = {(int(row.subject_number_norm), int(row.trial_id)): row for row in rounding.itertuples(index=False)}
    rows = []
    manifest = manifest.sort_values(['subject_number_norm', 'trial_id']).reset_index(drop=True)
    label_cache = {}
    for record in manifest.itertuples(index=False):
        subject_number = int(record.subject_number_norm)
        trial_id = int(record.trial_id)
        n_steps = int(record.label_steps)
        round_key = (subject_number, trial_id)
        if round_key not in rounding_lookup:
            raise REFEDSampleError(f'No rounding audit for subject={subject_number}, trial={trial_id}')
        rounding_record = rounding_lookup[round_key]
        delta = int(rounding_record.delta_samples)
        if delta not in {0, 1}:
            raise REFEDSampleError(f'Unexpected fNIRS trim for subject {subject_number}, trial {trial_id}: {delta}')
        expected_fnirs = n_steps * FNIRS_FS_DOWN
        if int(rounding_record.label_anchored_target) != expected_fnirs:
            raise REFEDSampleError('Rounding target mismatch')
        label_path = str(record.label_path)
        label_variable = str(record.label_variable)
        cache_key = (label_path, label_variable)
        if cache_key not in label_cache:
            mat = scipy.io.loadmat(label_path)
            if label_variable not in mat:
                raise REFEDSampleError(f'{label_path}: missing variable {label_variable}')
            label_cache[cache_key] = np.asarray(mat[label_variable])
        labels = label_cache[cache_key]
        if labels.shape != (n_steps, 2):
            raise REFEDSampleError(f'subject={subject_number}, trial={trial_id}: label shape {labels.shape}, expected {(n_steps, 2)}')
        valence_3c = label_to_3c_official(labels[:, 0])
        arousal_3c = label_to_3c_official(labels[:, 1])
        parent_id = str(record.trial_group_id)
        for sample_index in range(n_steps):
            sample_id = f'{parent_id}:s{sample_index:03d}'
            rows.append({'dataset': 'refed', 'preprocessing_protocol': PREPROCESSING_PROTOCOL, 'subject_id': str(record.subject_id), 'subject_number': subject_number, 'trial_id': trial_id, 'parent_trial_group_id': parent_id, 'sample_index': sample_index, 'sample_id': sample_id, 'time_step_sec': sample_index, 'valence_raw': float(labels[sample_index, 0]), 'arousal_raw': float(labels[sample_index, 1]), 'valence_3c': int(valence_3c[sample_index]), 'arousal_3c': int(arousal_3c[sample_index]), 'eeg_relative_path': f'data/{subject_number}/EEG_videos.mat', 'eeg_variable': f'video_{trial_id}', 'eeg_resampled_start': sample_index * EEG_FS_DOWN, 'eeg_resampled_end': (sample_index + 1) * EEG_FS_DOWN, 'fnirs_relative_path': f'data/{subject_number}/fNIRS_videos.mat', 'fnirs_variable': f'video_{trial_id}', 'fnirs_measurement_index': FNIRS_TSMMF_MEASUREMENT_INDEX, 'fnirs_resampled_start': sample_index * FNIRS_FS_DOWN, 'fnirs_resampled_end': (sample_index + 1) * FNIRS_FS_DOWN, 'fnirs_raw_samples': int(record.fnirs_samples), 'fnirs_resampled_samples_before_alignment': int(rounding_record.resampled_samples), 'fnirs_alignment_trim_samples': delta})
    samples = pd.DataFrame(rows)
    if samples.empty:
        raise REFEDSampleError('Empty sample registry')
    if samples['sample_id'].duplicated().any():
        raise REFEDSampleError('Duplicate sample_id')
    return samples
