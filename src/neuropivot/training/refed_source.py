from neuropivot.training.contracts import bind_anchor, assert_checkpoint_anchor
from neuropivot.paths import REPO, WORKDIR, OUTER_FOLD, OUTER_FOLDS, TRAIN_SUBJECTS_PUBLIC, N_TRAIN, N_TRIALS, fold_text
from pathlib import Path
import os
from neuropivot.data import refed_loader as base
import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
ROOT = REPO
from neuropivot.config import load_config
from neuropivot.models.eeg import Gate3EEGPivotV2, soft_kl
REFED_SAMPLE_REGISTRY = 'refed_1s_samples'
REFED_LOSO_PROTOCOL = 'refed_strict_loso_v1'
EAV_CACHE_PROTOCOL = 'gate3_eav_strict_speaking_windows_v1'
RUN_PROTOCOL = 'gate3a_refed_target_loso_v2'
SHUFFLED_MAPPING = [3, 0, 4, 1, 2]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def pearson_per_dim(true, pred):
    out = []
    for d in range(true.shape[1]):
        a = true[:, d]
        b = pred[:, d]
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            out.append(float('nan'))
        else:
            out.append(float(np.corrcoef(a, b)[0, 1]))
    return out

def ccc_1d(true, pred):
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    mt = float(true.mean())
    mp = float(pred.mean())
    vt = float(true.var())
    vp = float(pred.var())
    covariance = float(np.mean((true - mt) * (pred - mp)))
    denominator = vt + vp + (mt - mp) ** 2
    if denominator < 1e-12:
        return float('nan')
    return float(2.0 * covariance / denominator)

def ccc_per_dim(true, pred):
    return [ccc_1d(true[:, d], pred[:, d]) for d in range(true.shape[1])]

def regression_metrics(true, pred):
    error = np.abs(true - pred)
    return {'mae': float(error.mean()), 'mae_valence': float(error[:, 0].mean()), 'mae_arousal': float(error[:, 1].mean()), 'pearson': pearson_per_dim(true, pred), 'ccc': ccc_per_dim(true, pred)}

class REFEDVAData(Dataset):

    def __init__(self, eeg, va, mean, std):
        self.eeg = eeg
        self.va = va
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.va)

    def __getitem__(self, index):
        x = (self.eeg[index] - self.mean[0]) / self.std[0]
        return (torch.from_numpy(np.asarray(x, dtype=np.float32)), torch.from_numpy(np.asarray(self.va[index], dtype=np.float32)))

class EAVSourceData(Dataset):

    def __init__(self, eeg, labels, mean, std):
        self.eeg = eeg
        self.labels = labels
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = (self.eeg[index] - self.mean[0]) / self.std[0]
        return (torch.from_numpy(np.asarray(x, dtype=np.float32)), torch.tensor(int(self.labels[index]), dtype=torch.long))

def channel_stats(x):
    """
    x:
        [N,C,T]

    Return:
        [1,C,1]
    """
    mean = x.mean(axis=(0, 2))[None, :, None]
    std = x.std(axis=(0, 2))[None, :, None]
    std = np.maximum(std, 1e-06)
    return (mean.astype(np.float32), std.astype(np.float32))

def read_refed_loso_fold(cfg, test_subject):
    path = cfg.split_dir / f'{REFED_LOSO_PROTOCOL}.json'
    registry = json.loads(path.read_text(encoding='utf-8'))
    fold_id = f'test_subject_{test_subject:02d}'
    matches = [fold for fold in registry['folds'] if str(fold['fold_id']) == fold_id]
    if len(matches) != 1:
        raise RuntimeError(f'Could not resolve {fold_id}')
    fold = matches[0]
    train_subjects = [int(x) for x in fold['train_subject_ids']]
    val_subjects = [int(x) for x in fold['val_subject_ids']]
    test_subjects = [int(x) for x in fold['test_subject_ids']]
    if len(train_subjects) != 30:
        raise RuntimeError('Expected 30 REFED training subjects')
    if len(val_subjects) != 1:
        raise RuntimeError('Expected one REFED validation subject')
    if test_subjects != [test_subject]:
        raise RuntimeError('REFED test subject mismatch')
    if not (set(train_subjects).isdisjoint(val_subjects) and set(train_subjects).isdisjoint(test_subjects) and set(val_subjects).isdisjoint(test_subjects)):
        raise RuntimeError('REFED subject leakage')
    return {'fold_id': fold_id, 'train': train_subjects, 'val': val_subjects, 'test': test_subjects}

def load_refed_subject_group(cfg, sample_registry, subjects):
    eeg_parts = []
    va_parts = []
    subject_counts = {}
    for subject in subjects:
        eeg, fnirs, _, metadata, _, _ = base.load_subject_data(cfg, sample_registry, subject, 'valence')
        eeg = np.asarray(eeg, dtype=np.float32)
        if eeg.ndim == 4 and eeg.shape[1] == 1 and (eeg.shape[2] == 64):
            eeg = eeg[:, 0, :, :]
        elif eeg.ndim == 3 and eeg.shape[1] == 64:
            pass
        else:
            raise RuntimeError(f'subject {subject}: unsupported REFED EEG shape {eeg.shape}')
        raw_va = metadata[['valence_raw', 'arousal_raw']].to_numpy(dtype=np.float32)
        va = (raw_va - 1.0) / 254.0
        if eeg.ndim != 3 or eeg.shape[1] != 64:
            raise RuntimeError(f'subject {subject}: unexpected REFED EEG shape {eeg.shape}')
        if va.shape != (len(eeg), 2):
            raise RuntimeError(f'subject {subject}: VA shape mismatch')
        subject_counts[str(subject)] = int(len(eeg))
        print('  REFED subject', subject, 'EEG', eeg.shape, 'VA', va.shape)
        eeg_parts.append(eeg)
        va_parts.append(va)
        del fnirs
    return (np.concatenate(eeg_parts, axis=0), np.concatenate(va_parts, axis=0), subject_counts)

def refed_only_loss(model, output, true_va, lambda_semantic, sigma):
    va_loss = F.l1_loss(output['va_pred'], true_va)
    target = model.refed_targets(true_va, sigma=sigma)
    proto_kl, proto_ce, target_entropy = soft_kl(output['proto_logits'], target)
    total = va_loss + lambda_semantic * proto_kl
    return {'total': total, 'va': va_loss, 'proto_kl': proto_kl, 'proto_ce': proto_ce, 'target_entropy': target_entropy}

@torch.no_grad()
def evaluate_refed(model, loader, device, sigma):
    model.eval()
    true_all = []
    pred_all = []
    mae_sum = 0.0
    kl_sum = 0.0
    n = 0
    for x, va in loader:
        x = x.to(device, non_blocking=True)
        va = va.to(device, non_blocking=True)
        output = model.encode_refed(x)
        loss = refed_only_loss(model, output, va, lambda_semantic=1.0, sigma=sigma)
        bs = int(va.shape[0])
        n += bs
        mae_sum += float(loss['va']) * bs
        kl_sum += float(loss['proto_kl']) * bs
        true_all.append(va.cpu().numpy())
        pred_all.append(output['va_pred'].cpu().numpy())
    true = np.concatenate(true_all, axis=0)
    pred = np.concatenate(pred_all, axis=0)
    result = regression_metrics(true, pred)
    result['prototype_kl'] = kl_sum / n
    result['true'] = true
    result['pred'] = pred
    return result

def get_next(loader, iterator):
    try:
        batch = next(iterator)
        return (batch, iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
        return (batch, iterator)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths-config', type=Path, required=True)
    parser.add_argument('--mode', choices=['separate'], required=True)
    parser.add_argument('--refed-test-subject', type=int, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--lambda-semantic', type=float, default=0.25)
    parser.add_argument('--prototype-sigma', type=float, default=0.15)
    parser.add_argument('--max-steps-per-epoch', type=int, default=0)
    parser.add_argument('--run-tag', type=str, default='main')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Expected exactly one visible GPU')
    device = torch.device('cuda:0')
    set_seed(args.seed)
    cfg = load_config(args.paths_config)
    print('=' * 78)
    print('GATE3A REFED-TARGET STRICT LOSO')
    print('=' * 78)
    print('protocol:', RUN_PROTOCOL)
    print('mode:', 'separate')
    print('GPU:', torch.cuda.get_device_name(0))
    loso = read_refed_loso_fold(cfg, args.refed_test_subject)
    print('REFED fold:', loso['fold_id'])
    print('REFED train subjects:', loso['train'])
    print('REFED val subject:', loso['val'])
    print('REFED test subject:', loso['test'])
    sample_registry_path = cfg.split_dir / 'samples' / f'{REFED_SAMPLE_REGISTRY}.csv'
    sample_registry = pd.read_csv(sample_registry_path)
    print()
    print('Loading REFED TRAIN...')
    refed_train_eeg, refed_train_va, refed_train_counts = load_refed_subject_group(cfg, sample_registry, loso['train'])
    print()
    print('Loading REFED VAL...')
    refed_val_eeg, refed_val_va, refed_val_counts = load_refed_subject_group(cfg, sample_registry, loso['val'])
    print()
    print('Loading REFED TEST...')
    refed_test_eeg, refed_test_va, refed_test_counts = load_refed_subject_group(cfg, sample_registry, loso['test'])
    print()
    print('REFED shapes:')
    print('  train:', refed_train_eeg.shape, refed_train_va.shape)
    print('  val:', refed_val_eeg.shape, refed_val_va.shape)
    print('  test:', refed_test_eeg.shape, refed_test_va.shape)
    refed_mean, refed_std = channel_stats(refed_train_eeg)
    train_va_mean = refed_train_va.mean(axis=0).astype(np.float32)
    print()
    print('REFED train VA mean:', train_va_mean.tolist())
    eav_eeg = None
    eav_label = None
    eav_counts = None
    train_ds = REFEDVAData(refed_train_eeg, refed_train_va, refed_mean, refed_std)
    val_ds = REFEDVAData(refed_val_eeg, refed_val_va, refed_mean, refed_std)
    test_ds = REFEDVAData(refed_test_eeg, refed_test_va, refed_mean, refed_std)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    eav_loader = None
    model = Gate3EEGPivotV2().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    shuffled_mapping = None
    suffix = '' if args.run_tag == 'main' else '_' + args.run_tag
    out_dir = cfg.checkpoint_dir / 'gate3' / RUN_PROTOCOL / f"refed_test{args.refed_test_subject:02d}_{'separate'}_seed{args.seed:03d}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / 'best.pt'
    best_val_mae = float('inf')
    best_epoch = None
    stale = 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = {'total': 0.0, 'refed_va': 0.0, 'refed_kl': 0.0, 'eav_task': 0.0, 'eav_kl': 0.0}
        steps = len(train_loader)
        if args.max_steps_per_epoch > 0:
            steps = min(steps, args.max_steps_per_epoch)
        ref_iter = iter(train_loader)
        if eav_loader is not None:
            eav_iter = iter(eav_loader)
        else:
            eav_iter = None
        for _ in range(steps):
            ref_batch, ref_iter = get_next(train_loader, ref_iter)
            ref_x, ref_va = ref_batch
            ref_x = ref_x.to(device, non_blocking=True)
            ref_va = ref_va.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            ref_out = model.encode_refed(ref_x)
            ref_loss = refed_only_loss(model, ref_out, ref_va, lambda_semantic=args.lambda_semantic, sigma=args.prototype_sigma)
            total = ref_loss['total']
            total.backward()
            sums['total'] += float(total.detach())
            sums['refed_va'] += float(ref_loss['va'].detach())
            sums['refed_kl'] += float(ref_loss['proto_kl'].detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        for key in sums:
            sums[key] /= steps
        val = evaluate_refed(model, val_loader, device, sigma=args.prototype_sigma)
        updated = val['mae'] < best_val_mae
        if updated:
            best_val_mae = float(val['mae'])
            best_epoch = epoch
            stale = 0
            torch.save({'model_state': model.state_dict(), 'best_epoch': best_epoch, 'best_val_mae': best_val_mae, 'mode': 'separate', 'public_format_version': 1, 'model_config': {'embedding_dim': 128, 'shared_dim': 64, 'dropout': 0.25}, 'normalization': {'mean': torch.from_numpy(refed_mean), 'std': torch.from_numpy(refed_std), 'dataset': 'refed'}, 'refed_test_subject': args.refed_test_subject}, checkpoint)
        else:
            stale += 1
        print('epoch=%03d total=%.4f refVA=%.4f refKL=%.4f eavCE=%.4f eavKL=%.4f | valMAE=%.4f V=%.4f A=%.4f r=(%.3f,%.3f) ccc=(%.3f,%.3f) pat=%d' % (epoch, sums['total'], sums['refed_va'], sums['refed_kl'], sums['eav_task'], sums['eav_kl'], val['mae'], val['mae_valence'], val['mae_arousal'], val['pearson'][0], val['pearson'][1], val['ccc'][0], val['ccc'][1], stale))
        if stale >= args.patience:
            print('early stop:', epoch)
            break
    if not checkpoint.exists():
        raise RuntimeError('checkpoint missing')
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state'])
    test = evaluate_refed(model, test_loader, device, sigma=args.prototype_sigma)
    print()
    print('#' * 78)
    print('GATE3A REFED-TARGET STRICT LOSO TEST')
    print('#' * 78)
    print('mode:', 'separate')
    print('best epoch:', best_epoch)
    print('REFED TEST MAE:', test['mae'])
    print('REFED TEST V/A MAE:', [test['mae_valence'], test['mae_arousal']])
    print('REFED TEST Pearson:', test['pearson'])
    print('REFED TEST CCC:', test['ccc'])
    print('REFED TEST proto KL:', test['prototype_kl'])
    print()
    summary = {'run_protocol': RUN_PROTOCOL, 'mode': 'separate', 'seed': args.seed, 'strict_target_protocol': REFED_LOSO_PROTOCOL, 'refed_fold': loso, 'eav_source_protocol': None, 'eav_source_subjects': [], 'cross_dataset_subject_pairing': False, 'refed_test_used_for_training': False, 'refed_test_used_for_selection': False, 'refed_validation_used_for_selection': True, 'best_epoch': best_epoch, 'best_val_mae': best_val_mae, 'lambda_semantic': args.lambda_semantic, 'prototype_sigma': args.prototype_sigma, 'train_va_mean': train_va_mean.tolist(), 'test_metrics': {k: test[k] for k in ['mae', 'mae_valence', 'mae_arousal', 'pearson', 'ccc', 'prototype_kl']}, 'refed_train_subject_sample_counts': refed_train_counts, 'refed_val_subject_sample_counts': refed_val_counts, 'refed_test_subject_sample_counts': refed_test_counts, 'eav_source_subject_sample_counts': eav_counts, 'elapsed_sec': float(time.time() - start)}
    summary_path = out_dir / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=True) + '\n')
    np.savez(out_dir / 'test_predictions.npz', true=test['true'], pred=test['pred'], train_va_mean=train_va_mean)
    print()
    print('summary:', summary_path)
    print('\nGATE3A REFED-TARGET LOSO: PASS')
if __name__ == '__main__':
    main()
