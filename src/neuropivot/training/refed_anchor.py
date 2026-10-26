from neuropivot.training.contracts import bind_anchor, assert_checkpoint_anchor
from neuropivot.paths import REPO, WORKDIR, OUTER_FOLD, OUTER_FOLDS, TRAIN_SUBJECTS_PUBLIC, N_TRAIN, N_TRIALS, fold_text
from pathlib import Path
import os
from neuropivot.training import refed_source as v2
import argparse
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
ROOT = REPO
from neuropivot.config import load_config
from neuropivot.models.eeg import Gate3EEGPivotV2
PROTOCOL = 'gate3a_refed_anchor_v4'

@torch.no_grad()
def extract(model, loader, device):
    model.eval()
    shared = []
    true_va = []
    pred_va = []
    for x, va in loader:
        x = x.to(device, non_blocking=True)
        out = model.encode_refed(x)
        shared.append(out['shared'].cpu().numpy())
        pred_va.append(out['va_pred'].cpu().numpy())
        true_va.append(va.numpy())
    return (np.concatenate(shared, axis=0), np.concatenate(true_va, axis=0), np.concatenate(pred_va, axis=0))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths-config', type=Path, required=True)
    parser.add_argument('--refed-test-subject', type=int, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Expected exactly one visible GPU')
    device = torch.device('cuda:0')
    cfg = load_config(args.paths_config)
    anchor_contract = bind_anchor(cfg, args.refed_test_subject, args.seed)
    loso = v2.read_refed_loso_fold(cfg, args.refed_test_subject)
    registry = pd.read_csv(cfg.split_dir / 'samples' / f'{v2.REFED_SAMPLE_REGISTRY}.csv')
    print('REFED train:', loso['train'])
    print('REFED val:', loso['val'])
    print('REFED test:', loso['test'])
    print('\nLoading REFED train for TRAIN-ONLY statistics...')
    train_x, train_va, _ = v2.load_refed_subject_group(cfg, registry, loso['train'])
    mean, std = v2.channel_stats(train_x)
    train_va_mean = train_va.mean(axis=0).astype(np.float32)
    del train_x
    del train_va
    print('\nLoading REFED val...')
    val_x, val_va, _ = v2.load_refed_subject_group(cfg, registry, loso['val'])
    print('\nLoading REFED test...')
    test_x, test_va, _ = v2.load_refed_subject_group(cfg, registry, loso['test'])
    val_ds = v2.REFEDVAData(val_x, val_va, mean, std)
    test_ds = v2.REFEDVAData(test_x, test_va, mean, std)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    model = Gate3EEGPivotV2().to(device)
    checkpoint = cfg.checkpoint_dir / 'gate3' / 'gate3a_refed_target_loso_v2' / f'refed_test{args.refed_test_subject:02d}_separate_seed{args.seed:03d}_pilot' / 'best.pt'
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    if ckpt.get('refed_test_subject') != args.refed_test_subject:
        raise ValueError('Source checkpoint does not prove the requested REFED anchor split')
    model.load_state_dict(ckpt['model_state'])
    for p in model.parameters():
        p.requires_grad = False
    val_shared, val_true, val_pred = extract(model, val_loader, device)
    test_shared, test_true, test_pred = extract(model, test_loader, device)
    val_metrics = v2.regression_metrics(val_true, val_pred)
    test_metrics = v2.regression_metrics(test_true, test_pred)
    print()
    print('VAL:', val_metrics)
    print('DEV TEST:', test_metrics)
    out_dir = cfg.feature_dir / 'gate3' / PROTOCOL
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'refed_test{args.refed_test_subject:02d}_seed{args.seed:03d}.npz'
    np.savez(out_path, val_shared=val_shared.astype(np.float32), val_true_va=val_true.astype(np.float32), val_pred_va=val_pred.astype(np.float32), test_shared=test_shared.astype(np.float32), test_true_va=test_true.astype(np.float32), test_pred_va=test_pred.astype(np.float32), train_va_mean=train_va_mean)
    audit = {'protocol': PROTOCOL, 'refed_fold': loso, 'checkpoint': str(checkpoint), 'checkpoint_best_epoch': int(ckpt['best_epoch']), 'validation_metrics': val_metrics, 'dev_test_metrics': test_metrics, 'test_used_for_selection': False, 'cross_dataset_pairing': False, 'output': str(out_path)}
    audit_path = cfg.audit_dir / 'gate3' / f'{PROTOCOL}_refed_test{args.refed_test_subject:02d}.json'
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, allow_nan=True) + '\n')
    print()
    print('saved:', out_path)
    print('audit:', audit_path)
    print('\nGATE3A REFED ANCHOR V4: PASS')
if __name__ == '__main__':
    main()
