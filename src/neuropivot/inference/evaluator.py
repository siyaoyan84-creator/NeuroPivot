from neuropivot.paths import REPO, WORKDIR, OUTER_FOLD, OUTER_FOLDS, TRAIN_SUBJECTS_PUBLIC, N_TRAIN, N_TRIALS, fold_text
from pathlib import Path
import os

import argparse
import gc
import hashlib
import itertools
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
ROOT = REPO
SCRIPTS = ROOT / 'scripts'
from neuropivot.training import residual as ft
stageb = ft.stageb
CORRECT_EMOTIONS = ['neutral', 'anger', 'happiness', 'sadness', 'calmness']
stageb.EMOTIONS = list(CORRECT_EMOTIONS)
TRAIN_SUBJECTS = TRAIN_SUBJECTS_PUBLIC
VALIDATION_SUBJECT = int(ft.VALIDATION_SUBJECT)
OUTER_TEST_SUBJECTS = OUTER_FOLDS[OUTER_FOLD]
EXCLUDED_SUBJECTS = list(ft.PERMANENTLY_EXCLUDED_SUBJECTS)
PHYS_ROOT = Path(ft.PHYS_ROOT)
MODEL_ROOT = Path(ft.OUT_ROOT)
TASK_CKPT = Path(ft.TASK_CKPT)
RAW_OUT = WORKDIR / 'features' / 'gate4' / fold_text('gate4_grouped_outer01_evaluation_label_fix_v1') / fold_text('outer01_seed042.npz')
AUDIT_OUT = WORKDIR / 'audits' / 'gate4' / fold_text('gate4_grouped_outer01_label_fix_v1_seed042.json')

def load_torch(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(path, map_location='cpu')

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def as_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().astype(np.float32)
    return np.asarray(value, dtype=np.float32)

def load_condition(condition, representation, mode, feature_dim, shuffle_scope):
    path = MODEL_ROOT / f'{condition}_seed042' / 'final.pt'
    assert path.is_file(), path
    checkpoint = load_torch(path)
    expected_anchor=json.loads((WORKDIR/"checkpoints/gate3/neuropivot_anchor_contract.json").read_text(encoding="utf-8"))
    if checkpoint.get("anchor_contract")!=expected_anchor:
        raise ValueError("Residual checkpoint anchor contract is missing or mismatched")
    assert checkpoint['protocol'] == fold_text('gate4_grouped_outer01_residual_fit_label_fix_v1')
    assert checkpoint['condition'] == condition
    assert checkpoint['representation'] == representation
    assert checkpoint['mode'] == mode
    assert checkpoint['feature_dim'] == feature_dim
    assert checkpoint['shuffle_scope'] == shuffle_scope
    assert checkpoint['train_subjects'] == TRAIN_SUBJECTS
    assert checkpoint['train_trials'] == N_TRIALS
    assert checkpoint['validation_subject'] == VALIDATION_SUBJECT
    assert checkpoint['validation_subject_used_for_fit'] is False
    assert checkpoint['outer_test_subjects'] == OUTER_TEST_SUBJECTS
    assert checkpoint['outer_test_subjects_used_for_fit'] is False
    assert checkpoint['permanently_excluded_subjects'] == EXCLUDED_SUBJECTS
    assert checkpoint['subject01_used_for_fit'] is False
    assert checkpoint['epoch'] == 3
    assert checkpoint['seed'] == 42
    assert checkpoint['scientific_full_run'] is True
    return (path, checkpoint)

def prepare_rows(representation, checkpoint, shuffled, seed):
    rows = ft.load_rows(OUTER_TEST_SUBJECTS, representation, phys_root=PHYS_ROOT)
    n_trials = len(OUTER_TEST_SUBJECTS) * 100
    feature_dim = int(checkpoint['feature_dim'])
    assert len(rows) == n_trials
    x = np.stack([row['feature'] for row in rows], axis=0).astype(np.float32)
    assert x.shape == (n_trials, feature_dim)
    mean = as_numpy(checkpoint['feature_mean'])
    std = as_numpy(checkpoint['feature_std'])
    assert mean.shape == (feature_dim,)
    assert std.shape == (feature_dim,)
    assert np.isfinite(mean).all()
    assert np.isfinite(std).all()
    assert np.all(std > 1e-06)
    x_used = x.copy()
    source_index = np.arange(n_trials, dtype=np.int64)
    z = ((x_used - mean) / std).astype(np.float32)
    assert np.isfinite(z).all()
    for i, row in enumerate(rows):
        row['z'] = z[i].copy()
    return (rows, source_index)

def make_batch(row):
    return {'label': torch.tensor([row['label']], dtype=torch.long), 'z': torch.from_numpy(row['z']).float().unsqueeze(0), 'audio': torch.from_numpy(row['audio']).unsqueeze(0), 'frame': torch.from_numpy(row['frame']).unsqueeze(0), 'multi': torch.from_numpy(row['multi']).unsqueeze(0)}

@torch.inference_mode()
def true_nll(model, builder, row, use_residual):
    embeds, labels, attention, _, _ = builder.build(make_batch(row), use_residual=use_residual)
    output = model.llama_model(inputs_embeds=embeds, attention_mask=attention, labels=labels, return_dict=True, use_cache=False)
    value = float(output.loss.detach().cpu())
    assert np.isfinite(value)
    return value
