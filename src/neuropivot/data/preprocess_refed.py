from neuropivot.paths import REPO, WORKDIR, OUTER_FOLD, OUTER_FOLDS, TRAIN_SUBJECTS_PUBLIC, N_TRAIN, N_TRIALS, fold_text
from pathlib import Path
import os

import argparse
import gc
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io
PROJECT_ROOT = REPO
from neuropivot.config import load_config
from neuropivot.preprocessing.refed import preprocess_refed_eeg_official, preprocess_refed_fnirs_official
PROTOCOL = 'refed_official_1s_preproc_v1'

def atomic_savemat(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + '.tmp')
    try:
        scipy.io.savemat(tmp, payload, appendmat=False, do_compression=False)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()

def process_subject(subject_number, subject_manifest, sample_registry, output_root):
    subject_manifest = subject_manifest.sort_values('trial_id').reset_index(drop=True)
    if len(subject_manifest) != 15:
        raise RuntimeError(f'subject {subject_number}: expected 15 trials, got {len(subject_manifest)}')
    subject_samples = sample_registry[sample_registry['subject_number'] == subject_number]
    if len(subject_samples) != 1535:
        raise RuntimeError(f'subject {subject_number}: expected 1535 registry samples, got {len(subject_samples)}')
    subject_dir = output_root / str(subject_number)
    subject_dir.mkdir(parents=True, exist_ok=True)
    audit_rows = []
    eeg_paths = subject_manifest['eeg_video_path'].unique().tolist()
    if len(eeg_paths) != 1:
        raise RuntimeError(f'subject {subject_number}: expected one EEG MAT, got {eeg_paths}')
    eeg_source_path = Path(eeg_paths[0])
    eeg_raw_mat = scipy.io.loadmat(eeg_source_path)
    eeg_output = {}
    for row in subject_manifest.itertuples(index=False):
        trial_id = int(row.trial_id)
        variable = str(row.eeg_video_variable)
        raw = np.asarray(eeg_raw_mat[variable])
        samples = preprocess_refed_eeg_official(raw)
        expected_n = int(row.label_steps)
        if samples.shape != (expected_n, 64, 200):
            raise RuntimeError(f'subject {subject_number}, trial {trial_id}: EEG shape {samples.shape}')
        eeg_output[f'video_{trial_id}'] = samples
    eeg_output_path = subject_dir / 'EEG_videos.mat'
    atomic_savemat(eeg_output_path, eeg_output)
    del eeg_raw_mat
    del eeg_output
    gc.collect()
    fnirs_paths = subject_manifest['fnirs_video_path'].unique().tolist()
    if len(fnirs_paths) != 1:
        raise RuntimeError(f'subject {subject_number}: expected one fNIRS MAT, got {fnirs_paths}')
    fnirs_source_path = Path(fnirs_paths[0])
    fnirs_raw_mat = scipy.io.loadmat(fnirs_source_path)
    fnirs_output = {}
    for row in subject_manifest.itertuples(index=False):
        trial_id = int(row.trial_id)
        variable = str(row.fnirs_video_variable)
        raw = np.asarray(fnirs_raw_mat[variable])
        samples, metadata = preprocess_refed_fnirs_official(raw)
        expected_n = int(row.label_steps)
        if samples.shape != (expected_n, 6, 51, 25):
            raise RuntimeError(f'subject {subject_number}, trial {trial_id}: fNIRS shape {samples.shape}')
        parent_id = str(row.trial_group_id)
        parent_registry = subject_samples[subject_samples['parent_trial_group_id'] == parent_id]
        if len(parent_registry) != expected_n:
            raise RuntimeError(f'{parent_id}: registry count mismatch')
        trim_values = parent_registry['fnirs_alignment_trim_samples'].unique().tolist()
        if len(trim_values) != 1:
            raise RuntimeError(f'{parent_id}: inconsistent registry trim values')
        registry_trim = int(trim_values[0])
        actual_trim = int(metadata['trim_samples'])
        if registry_trim != actual_trim:
            raise RuntimeError(f'{parent_id}: trim mismatch registry={registry_trim}, preprocessing={actual_trim}')
        fnirs_output[f'video_{trial_id}'] = samples
        audit_rows.append({'preprocessing_protocol': PROTOCOL, 'subject_number': subject_number, 'trial_id': trial_id, 'parent_trial_group_id': parent_id, 'label_steps': expected_n, 'eeg_windows': expected_n, 'fnirs_windows': int(samples.shape[0]), 'fnirs_raw_samples': int(metadata['raw_samples']), 'fnirs_resampled_samples_before_trim': int(metadata['resampled_samples_before_trim']), 'fnirs_trim_samples': actual_trim, 'fnirs_resampled_samples_after_trim': int(metadata['resampled_samples_after_trim'])})
    fnirs_output_path = subject_dir / 'fNIRS_videos.mat'
    atomic_savemat(fnirs_output_path, fnirs_output)
    del fnirs_raw_mat
    del fnirs_output
    gc.collect()
    print(f'subject {subject_number:02d}: PASS | samples=1535')
    return audit_rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths-config', required=True, type=Path)
    parser.add_argument('--subjects', nargs='*', type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.paths_config)
    manifest_path = config.manifest_dir / 'refed_manifest.csv'
    sample_registry_path = config.split_dir / 'samples' / 'refed_1s_samples.csv'
    manifest = pd.read_csv(manifest_path)
    samples = pd.read_csv(sample_registry_path)
    output_root = config.feature_dir / 'refed_official_1s_preprocessed'
    output_root.mkdir(parents=True, exist_ok=True)
    if args.subjects:
        subjects = sorted(set(args.subjects))
    else:
        subjects = list(range(1, 33))
    for subject_number in subjects:
        if not 1 <= subject_number <= 32:
            raise RuntimeError(f'invalid subject: {subject_number}')
    audit_rows = []
    for subject_number in subjects:
        subject_manifest = manifest[manifest['subject_number'] == subject_number]
        rows = process_subject(subject_number, subject_manifest, samples, output_root)
        audit_rows.extend(rows)
    audit = pd.DataFrame(audit_rows)
    audit_dir = config.audit_dir / 'gate1'
    audit_dir.mkdir(parents=True, exist_ok=True)
    if len(subjects) == 32:
        audit_name = 'refed_official_preprocessed_cache_audit.csv'
    else:
        tag = '_'.join((str(x) for x in subjects))
        audit_name = f'refed_official_preprocessed_cache_subjects_{tag}_audit.csv'
    audit_path = audit_dir / audit_name
    audit.to_csv(audit_path, index=False)
    print()
    print('subjects processed:', len(subjects))
    print('trials processed:', len(audit))
    print('total windows:', int(audit['label_steps'].sum()))
    print('trim distribution:', audit['fnirs_trim_samples'].value_counts().sort_index().to_dict())
    print('cache root:', output_root)
    print('audit:', audit_path)
    print('\nREFED OFFICIAL PREPROCESSED CACHE: PASS')
if __name__ == '__main__':
    main()
