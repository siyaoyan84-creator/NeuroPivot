from neuropivot.paths import REPO, WORKDIR, OUTER_FOLD, OUTER_FOLDS, TRAIN_SUBJECTS_PUBLIC, N_TRAIN, N_TRIALS, fold_text
from pathlib import Path
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from neuropivot.config import load_config
from neuropivot.models.ordinal import Gate3EEGPivotV3
from neuropivot.training import bridge_base as v4
PIVOT_DIM = 64

def load_window_cache(cfg, subject):
    p = cfg.feature_dir / 'gate3' / v4.v2.EAV_CACHE_PROTOCOL / f'subject{subject:02d}.npz'
    if not p.is_file():
        raise FileNotFoundError(p)
    z = np.load(p, allow_pickle=True)
    required = ['eeg', 'label', 'sequence_id', 'window_index', 'parent_trial_group_id']
    missing = [k for k in required if k not in z.files]
    if missing:
        raise RuntimeError(f'{p}: missing {missing}')
    out = {'path': p, 'eeg': z['eeg'].astype(np.float32), 'label': z['label'].astype(np.int64), 'sequence_id': z['sequence_id'].astype(np.int64), 'window_index': z['window_index'].astype(np.int64), 'parent_trial_group_id': z['parent_trial_group_id'].astype(str)}
    if out['eeg'].shape != (400, 30, 500):
        raise RuntimeError(f"subject{subject:02d}: bad EEG shape {out['eeg'].shape}")
    for k in ['label', 'sequence_id', 'window_index', 'parent_trial_group_id']:
        if out[k].shape != (400,):
            raise RuntimeError(f'subject{subject:02d}: {k} shape={out[k].shape}')
    return out

@torch.no_grad()
def predict_windows(model, cache, mean, std, batch_size, device):
    ds = v4.v2.EAVSourceData(cache['eeg'], cache['label'], mean, std)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    va_parts = []
    shared_parts = []
    model.eval()
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        out = model.encode_eav(x)
        va = out['va_pred']
        shared = out['shared']
        if va.ndim != 2 or va.shape[1] != 2:
            raise RuntimeError(f'unexpected va_pred shape: {tuple(va.shape)}')
        if shared.ndim != 2 or shared.shape[1] != PIVOT_DIM:
            raise RuntimeError(f'unexpected shared shape: {tuple(shared.shape)}')
        va_parts.append(va.detach().cpu().float().numpy())
        shared_parts.append(shared.detach().cpu().float().numpy())
    va = np.concatenate(va_parts, axis=0).astype(np.float32, copy=False)
    shared = np.concatenate(shared_parts, axis=0).astype(np.float32, copy=False)
    if va.shape != (400, 2):
        raise RuntimeError(f'bad window VA shape {va.shape}')
    if shared.shape != (400, PIVOT_DIM):
        raise RuntimeError(f'bad window shared shape {shared.shape}')
    if not np.isfinite(va).all():
        raise RuntimeError('window VA contains NaN/Inf')
    if not np.isfinite(shared).all():
        raise RuntimeError('window shared contains NaN/Inf')
    return (va, shared)

def aggregate_trials(cache, window_va, window_shared):
    seq = cache['sequence_id']
    labels = cache['label']
    win = cache['window_index']
    parent = cache['parent_trial_group_id']
    unique_seq = []
    seen = set()
    for x in seq.tolist():
        x = int(x)
        if x not in seen:
            seen.add(x)
            unique_seq.append(x)
    if len(unique_seq) != 100:
        raise RuntimeError(f'expected 100 speaking trials, got {len(unique_seq)}')
    trial_va = []
    trial_va_std = []
    trial_shared = []
    trial_shared_std = []
    trial_label = []
    trial_parent = []
    for s in unique_seq:
        idx = np.where(seq == s)[0]
        if len(idx) != 4:
            raise RuntimeError(f'sequence {s}: expected 4 windows, got {len(idx)}')
        windows = sorted(win[idx].tolist())
        if windows != [0, 1, 2, 3]:
            raise RuntimeError(f'sequence {s}: bad windows {windows}')
        labels_s = np.unique(labels[idx])
        parent_s = np.unique(parent[idx])
        if len(labels_s) != 1:
            raise RuntimeError(f'sequence {s}: label mismatch')
        if len(parent_s) != 1:
            raise RuntimeError(f'sequence {s}: parent mismatch')
        trial_va.append(window_va[idx].mean(axis=0))
        trial_va_std.append(window_va[idx].std(axis=0))
        trial_shared.append(window_shared[idx].mean(axis=0))
        trial_shared_std.append(window_shared[idx].std(axis=0))
        trial_label.append(int(labels_s[0]))
        trial_parent.append(str(parent_s[0]))
    out = {'sequence_id': np.asarray(unique_seq, dtype=np.int64), 'label': np.asarray(trial_label, dtype=np.int64), 'parent_trial_group_id': np.asarray(trial_parent, dtype=str), 'teacher_va': np.asarray(trial_va, dtype=np.float32), 'teacher_window_std': np.asarray(trial_va_std, dtype=np.float32), 'pivot_shared': np.asarray(trial_shared, dtype=np.float32), 'pivot_window_std': np.asarray(trial_shared_std, dtype=np.float32)}
    expected = {'sequence_id': (100,), 'label': (100,), 'parent_trial_group_id': (100,), 'teacher_va': (100, 2), 'teacher_window_std': (100, 2), 'pivot_shared': (100, PIVOT_DIM), 'pivot_window_std': (100, PIVOT_DIM)}
    for k, shape in expected.items():
        if out[k].shape != shape:
            raise RuntimeError(f'{k}: shape={out[k].shape}, expected={shape}')
    return out
