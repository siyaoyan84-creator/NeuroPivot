from neuropivot.paths import REPO, WORKDIR, OUTER_FOLD, OUTER_FOLDS, TRAIN_SUBJECTS_PUBLIC, N_TRAIN, N_TRIALS, fold_text
from pathlib import Path
import os

from collections import Counter
from pathlib import Path
import mne
import numpy as np
import pandas as pd
MANIFEST = Path(str(WORKDIR / 'manifests' / 'refed_manifest.csv'))
TARGET_FS = 25
NOMINAL_FS = 1000 / 21

def main():
    df = pd.read_csv(MANIFEST)
    required = {'subject_id', 'trial_id', 'fnirs_samples', 'label_steps'}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f'Missing manifest columns: {sorted(missing)}')
    assert len(df) == 480
    unique_lengths = sorted(df['fnirs_samples'].astype(int).unique().tolist())
    output_length = {}
    for n_raw in unique_lengths:
        dummy = np.zeros(n_raw, dtype=np.float64)
        y = mne.filter.resample(dummy, up=TARGET_FS, down=NOMINAL_FS, verbose=False)
        output_length[n_raw] = int(y.shape[-1])
    rows = []
    for row in df.itertuples(index=False):
        n_raw = int(row.fnirs_samples)
        n_label = int(row.label_steps)
        n_resampled = output_length[n_raw]
        target = n_label * TARGET_FS
        delta = n_resampled - target
        rows.append({'subject_id': row.subject_id, 'trial_id': int(row.trial_id), 'fnirs_raw_samples': n_raw, 'label_steps': n_label, 'resampled_samples': n_resampled, 'label_anchored_target': target, 'delta_samples': delta, 'delta_seconds': delta / TARGET_FS})
    audit = pd.DataFrame(rows)
    print('rows:', len(audit))
    print('subjects:', audit['subject_id'].nunique())
    print('unique raw lengths:', len(unique_lengths))
    print('\n=== DELTA DISTRIBUTION ===')
    print(audit['delta_samples'].value_counts().sort_index().to_string())
    print('\nmin delta:', audit['delta_samples'].min())
    print('max delta:', audit['delta_samples'].max())
    bad = audit[(audit['delta_samples'] < 0) | (audit['delta_samples'] > 1)]
    nonzero = audit[audit['delta_samples'] != 0]
    print('\nnonzero trials:', len(nonzero))
    if len(nonzero):
        print(nonzero[['subject_id', 'trial_id', 'fnirs_raw_samples', 'label_steps', 'resampled_samples', 'label_anchored_target', 'delta_samples']].head(100).to_string(index=False))
    output = WORKDIR / 'audits/gate1/refed_fnirs_resample_rounding.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output, index=False)
    if len(bad):
        print('\n=== UNEXPECTED DELTAS ===')
        print(bad.to_string(index=False))
        raise RuntimeError('Found fNIRS resample delta outside {0, +1}')
    print('\nsaved:', output)
    print('\nREFED fNIRS RESAMPLE ROUNDING AUDIT: PASS')
if __name__ == '__main__':
    main()
