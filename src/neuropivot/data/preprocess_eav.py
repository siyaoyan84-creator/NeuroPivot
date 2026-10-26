from neuropivot.paths import REPO, WORKDIR, OUTER_FOLD, OUTER_FOLDS, TRAIN_SUBJECTS_PUBLIC, N_TRAIN, N_TRIALS, fold_text
from pathlib import Path
import os

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from neuropivot.config import load_config
from neuropivot.preprocessing.eav import load_eav_eeg, preprocess_eav_subject, segment_eav_trials
PREPROCESSING = 'eav_strict_pertrial_preproc_v1'
CACHE_PROTOCOL = 'gate3_eav_strict_speaking_windows_v1'
WINDOWS_PER_TRIAL = 4
EMOTION_TO_ID = {'neutral': 0, 'anger': 1, 'happiness': 2, 'sadness': 3, 'calmness': 4}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths-config', type=Path, required=True)
    parser.add_argument('--subjects', type=int, nargs='+', default=None)
    args = parser.parse_args()
    cfg = load_config(args.paths_config)
    manifest_path = cfg.manifest_dir / 'eav_manifest.csv'
    manifest = pd.read_csv(manifest_path)
    subjects = args.subjects if args.subjects is not None else sorted(manifest['subject_number'].unique().tolist())
    out_dir = cfg.feature_dir / 'gate3' / CACHE_PROTOCOL
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for subject in subjects:
        rows = manifest[(manifest['subject_number'] == subject) & (manifest['condition'].astype(str).str.lower() == 'speaking')].copy()
        rows = rows.sort_values('eeg_trial_index').reset_index(drop=True)
        if len(rows) != 100:
            raise RuntimeError(f'subject {subject}: expected 100 speaking trials, got {len(rows)}')
        if rows['eeg_path'].nunique() != 1:
            raise RuntimeError(f'multiple EEG files for subject {subject}')
        if rows['eeg_variable'].nunique() != 1:
            raise RuntimeError(f'multiple EEG variables for subject {subject}')
        eeg_path = rows['eeg_path'].iloc[0]
        eeg_variable = rows['eeg_variable'].iloc[0]
        print()
        print('=' * 78)
        print('subject:', subject)
        print('EEG:', eeg_path)
        print('variable:', eeg_variable)
        raw = load_eav_eeg(eeg_path, eeg_variable)
        print('raw:', raw.shape)
        trials = preprocess_eav_subject(raw, PREPROCESSING)
        if trials.shape != (200, 2000, 30):
            raise RuntimeError(f'unexpected preprocessed shape {trials.shape}')
        trial_indices = rows['eeg_trial_index'].to_numpy(dtype=np.int64)
        speaking_trials = trials[trial_indices]
        windows = segment_eav_trials(speaking_trials)
        if windows.shape != (400, 30, 500, 1):
            raise RuntimeError(f'unexpected window shape {windows.shape}')
        windows = windows[..., 0].astype(np.float32, copy=False)
        emotions = rows['emotion'].astype(str).str.lower().map(EMOTION_TO_ID)
        if emotions.isna().any():
            raise RuntimeError('unknown EAV emotion')
        labels = np.repeat(emotions.to_numpy(dtype=np.int64), WINDOWS_PER_TRIAL)
        parent_ids = np.repeat(rows['trial_group_id'].astype(str).to_numpy(), WINDOWS_PER_TRIAL)
        sequence_ids = np.repeat(rows['sequence_id'].to_numpy(dtype=np.int64), WINDOWS_PER_TRIAL)
        window_index = np.tile(np.arange(WINDOWS_PER_TRIAL, dtype=np.int64), len(rows))
        window_id = np.asarray([f'{parent}:w{int(w):02d}' for parent, w in zip(parent_ids, window_index)])
        expected_counts = [80, 80, 80, 80, 80]
        actual_counts = np.bincount(labels, minlength=5).tolist()
        if actual_counts != expected_counts:
            raise RuntimeError(f'subject {subject}: class counts {actual_counts}')
        out_path = out_dir / f'subject{subject:02d}.npz'
        np.savez(out_path, eeg=windows, label=labels, window_id=window_id, parent_trial_group_id=parent_ids, sequence_id=sequence_ids, window_index=window_index, subject_number=np.full(len(labels), subject, dtype=np.int64))
        size_mb = out_path.stat().st_size / 1024 / 1024
        print('windows:', windows.shape)
        print('labels:', actual_counts)
        print('saved:', out_path)
        print('size MB:', round(size_mb, 2))
        summary.append({'subject': int(subject), 'windows': int(len(labels)), 'class_counts': actual_counts, 'path': str(out_path)})
    audit = {'protocol': CACHE_PROTOCOL, 'preprocessing': PREPROCESSING, 'subjects': subjects, 'n_subjects': len(subjects), 'windows_per_subject': 400, 'channels': 30, 'samples_per_window': 500, 'emotion_to_id': EMOTION_TO_ID, 'records': summary}
    audit_path = cfg.audit_dir / 'gate3' / f'{CACHE_PROTOCOL}_audit.json'
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2) + '\n')
    print()
    print('audit:', audit_path)
    print('\nGATE3 EAV EEG CACHE: PASS')
if __name__ == '__main__':
    main()
