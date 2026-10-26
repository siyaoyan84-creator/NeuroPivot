from neuropivot.paths import REPO, WORKDIR, OUTER_FOLD, OUTER_FOLDS, TRAIN_SUBJECTS_PUBLIC, N_TRAIN, N_TRIALS, fold_text
from pathlib import Path
import os

import argparse
import json
import sys
from pathlib import Path
import pandas as pd
PROJECT_ROOT = REPO
from neuropivot.config import load_config
from neuropivot.samples.refed import build_refed_sample_registry

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths-config', required=True, type=Path)
    args = parser.parse_args()
    config = load_config(args.paths_config)
    manifest_path = config.manifest_dir / 'refed_manifest.csv'
    rounding_path = config.audit_dir / 'gate1' / 'refed_fnirs_resample_rounding.csv'
    manifest = pd.read_csv(manifest_path)
    rounding = pd.read_csv(rounding_path)
    samples = build_refed_sample_registry(manifest, rounding)
    output_dir = config.split_dir / 'samples'
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / 'refed_1s_samples.csv'
    samples.to_csv(output, index=False)
    subject_counts = samples.groupby('subject_number').size()
    trial_counts = samples.groupby(['subject_number', 'trial_id']).size()
    trim_trials = samples.groupby(['subject_number', 'trial_id'])['fnirs_alignment_trim_samples'].first()
    print('rows:', len(samples))
    print('subjects:', samples['subject_number'].nunique())
    print('parent trials:', samples['parent_trial_group_id'].nunique())
    print('unique samples:', samples['sample_id'].nunique())
    print('samples/subject min/max:', int(subject_counts.min()), int(subject_counts.max()))
    print('trimmed parent trials:', int((trim_trials == 1).sum()))
    print('\nvalence 3c:', samples['valence_3c'].value_counts().sort_index().to_dict())
    print('arousal 3c:', samples['arousal_3c'].value_counts().sort_index().to_dict())
    print('\nsaved:', output)
    print('\nREFED 1S SAMPLE REGISTRY: PASS')
if __name__ == '__main__':
    main()
