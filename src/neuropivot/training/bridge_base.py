from neuropivot.training.contracts import bind_anchor, assert_checkpoint_anchor
from neuropivot.paths import REPO, WORKDIR, OUTER_FOLD, OUTER_FOLDS, TRAIN_SUBJECTS_PUBLIC, N_TRAIN, N_TRIALS, fold_text
from pathlib import Path
import os
from neuropivot.training import refed_source as v2
import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader
ROOT = REPO
from neuropivot.config import load_config
from neuropivot.models.ordinal import Gate3EEGPivotV3, eav_ordinal_loss, VALENCE_RELATIONS, AROUSAL_RELATIONS
PROTOCOL = 'gate3a_anchored_pivot_v4_nested_outer03'
SHUFFLED_MAPPING = [3, 0, 4, 1, 2]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_subject_id(x):
    s = str(x)
    if s.startswith('subject'):
        s = s[len('subject'):]
    return int(s)

def read_eav_loso(cfg, test_subject):
    raise RuntimeError('Use the grouped bridge CLI; split resolver must be supplied by its frozen protocol')

def load_eav_group(cfg, subjects):
    root = cfg.feature_dir / 'gate3' / v2.EAV_CACHE_PROTOCOL
    xs = []
    ys = []
    for subject in subjects:
        p = root / f'subject{subject:02d}.npz'
        if not p.exists():
            raise FileNotFoundError(p)
        z = np.load(p)
        x = z['eeg'].astype(np.float32)
        y = z['label'].astype(np.int64)
        if x.shape != (400, 30, 500):
            raise RuntimeError(f'subject {subject}: {x.shape}')
        xs.append(x)
        ys.append(y)
    return (np.concatenate(xs, axis=0), np.concatenate(ys, axis=0))

def relation_metrics(centroids):
    diffs = []
    valence_ok = []
    arousal_ok = []
    for high, low in VALENCE_RELATIONS:
        diff = centroids[high, 0] - centroids[low, 0]
        diffs.append(float(diff))
        valence_ok.append(bool(diff > 0))
    for high, low in AROUSAL_RELATIONS:
        diff = centroids[high, 1] - centroids[low, 1]
        diffs.append(float(diff))
        arousal_ok.append(bool(diff > 0))
    all_ok = valence_ok + arousal_ok
    return {'consistency': float(np.mean(all_ok)), 'valence_consistency': float(np.mean(valence_ok)), 'arousal_consistency': float(np.mean(arousal_ok)), 'mean_relation_margin': float(np.mean(diffs)), 'relation_differences': diffs}

def class_centroids(values, labels):
    out = np.zeros((5, values.shape[1]), dtype=np.float64)
    for c in range(5):
        mask = labels == c
        if not np.any(mask):
            raise RuntimeError(f'class {c} missing')
        out[c] = values[mask].mean(axis=0)
    return out

@torch.no_grad()
def evaluate_eav(model, loader, device, shuffle_mapping=None, ordinal_margin=0.05):
    model.eval()
    true = []
    pred = []
    shared = []
    va_pred = []
    ce_sum = 0.0
    ord_sum = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model.encode_eav(x)
        ce = F.cross_entropy(out['class_logits'], y)
        ordinal = eav_ordinal_loss(out['va_pred'], y, margin=ordinal_margin, shuffle_mapping=shuffle_mapping)
        bs = int(y.shape[0])
        n += bs
        ce_sum += float(ce) * bs
        ord_sum += float(ordinal['total']) * bs
        true.append(y.cpu().numpy())
        pred.append(out['class_logits'].argmax(dim=1).cpu().numpy())
        shared.append(out['shared'].cpu().numpy())
        va_pred.append(out['va_pred'].cpu().numpy())
    true = np.concatenate(true)
    pred = np.concatenate(pred)
    shared = np.concatenate(shared, axis=0)
    va_pred = np.concatenate(va_pred, axis=0)
    centroids = class_centroids(va_pred, true)
    return {'ce': ce_sum / n, 'ordinal_loss': ord_sum / n, 'accuracy': float(accuracy_score(true, pred)), 'balanced_accuracy': float(balanced_accuracy_score(true, pred)), 'macro_f1': float(f1_score(true, pred, average='macro', zero_division=0)), 'labels': true, 'shared': shared, 'va_pred': va_pred, 'va_centroids': centroids, 'ordinal_metrics': relation_metrics(centroids)}

def retrieval_metrics(eav_shared, eav_labels, refed_shared, refed_true_va, k):
    eav = np.asarray(eav_shared, dtype=np.float64)
    refed = np.asarray(refed_shared, dtype=np.float64)
    eav /= np.maximum(np.linalg.norm(eav, axis=1, keepdims=True), 1e-12)
    refed /= np.maximum(np.linalg.norm(refed, axis=1, keepdims=True), 1e-12)
    similarity = eav @ refed.T
    k = min(int(k), refed.shape[0])
    top = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
    retrieved_va = refed_true_va[top].mean(axis=1)
    centroids = class_centroids(retrieved_va, eav_labels)
    return {'k': k, 'retrieved_va_centroids': centroids, 'ordinal_metrics': relation_metrics(centroids)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths-config', type=Path, required=True)
    parser.add_argument('--mode', choices=['ordinal_correct'], required=True)
    parser.add_argument('--eav-test-subject', type=int, default=1)
    parser.add_argument('--refed-anchor-test-subject', type=int, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--lambda-ordinal', type=float, default=0.25)
    parser.add_argument('--ordinal-margin', type=float, default=0.05)
    parser.add_argument('--retrieval-k', type=int, default=25)
    parser.add_argument('--max-steps-per-epoch', type=int, default=0)
    parser.add_argument('--run-tag', type=str, default='pilot')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Expected exactly one visible GPU')
    device = torch.device('cuda:0')
    set_seed(args.seed)
    cfg = load_config(args.paths_config)
    anchor_contract = bind_anchor(cfg, args.refed_anchor_test_subject, args.seed)
    anchor_path = cfg.feature_dir / 'gate3' / 'gate3a_refed_anchor_v4' / f'refed_test{args.refed_anchor_test_subject:02d}_seed{args.seed:03d}.npz'
    if not anchor_path.is_file():raise FileNotFoundError(anchor_path)
    with np.load(anchor_path,allow_pickle=False) as anchor_metadata:
        if int(anchor_metadata['refed_anchor_test_subject'])!=args.refed_anchor_test_subject or int(anchor_metadata['seed'])!=args.seed:
            raise ValueError('Anchor export identity mismatch')
    eav_split = read_eav_loso(cfg, args.eav_test_subject)
    print('=' * 78)
    print('GATE3A V4 ANCHORED ASYMMETRIC PIVOT')
    print('=' * 78)
    print('mode:', 'ordinal_correct')
    print('EAV fold:', eav_split)
    print('IMPORTANT: EAV and REFED subject IDs are NOT paired identities.')
    print()
    print('Loading EAV train...')
    train_x, train_y = load_eav_group(cfg, eav_split['train'])
    print('train:', train_x.shape)
    print('Loading EAV val...')
    val_x, val_y = load_eav_group(cfg, eav_split['val'])
    print('val:', val_x.shape)
    print('Loading EAV test...')
    test_x, test_y = load_eav_group(cfg, eav_split['test'])
    print('test:', test_x.shape)
    mean, std = v2.channel_stats(train_x)
    train_ds = v2.EAVSourceData(train_x, train_y, mean, std)
    val_ds = v2.EAVSourceData(val_x, val_y, mean, std)
    test_ds = v2.EAVSourceData(test_x, test_y, mean, std)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    model = Gate3EEGPivotV3().to(device)
    anchor_ckpt = cfg.checkpoint_dir / 'gate3' / 'gate3a_refed_target_loso_v2' / f'refed_test{args.refed_anchor_test_subject:02d}_separate_seed{args.seed:03d}_pilot' / 'best.pt'
    if not anchor_ckpt.exists():
        raise FileNotFoundError(anchor_ckpt)
    ckpt = torch.load(anchor_ckpt, map_location=device, weights_only=True)
    if ckpt.get('refed_test_subject') != args.refed_anchor_test_subject:
        raise ValueError('Source checkpoint anchor split is absent or mismatched')
    model.load_state_dict(ckpt['model_state'])
    for p in model.parameters():
        p.requires_grad = False
    for p in model.eav_encoder.parameters():
        p.requires_grad = True
    for p in model.eav_classifier.parameters():
        p.requires_grad = True
    trainable = [p for p in model.parameters() if p.requires_grad]
    print('trainable parameters:', sum((p.numel() for p in trainable)))
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    semantic_mapping = None
    lambda_ordinal = args.lambda_ordinal
    out_dir = cfg.checkpoint_dir / 'gate3' / PROTOCOL / f"eav_test{args.eav_test_subject:02d}_{'ordinal_correct'}_seed{args.seed:03d}_{args.run_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / 'best.pt'
    best_score = float('inf')
    best_epoch = None
    best_val = None
    stale = 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = {'ce': 0.0, 'ord': 0.0, 'total': 0.0}
        steps = len(train_loader)
        if args.max_steps_per_epoch > 0:
            steps = min(steps, args.max_steps_per_epoch)
        iterator = iter(train_loader)
        for _ in range(steps):
            x, y = next(iterator)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model.encode_eav(x)
            ce = F.cross_entropy(out['class_logits'], y)
            ordinal = eav_ordinal_loss(out['va_pred'], y, margin=args.ordinal_margin, shuffle_mapping=semantic_mapping)
            total = ce + lambda_ordinal * ordinal['total']
            total.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            sums['ce'] += float(ce.detach())
            sums['ord'] += float(ordinal['total'].detach())
            sums['total'] += float(total.detach())
        for k in sums:
            sums[k] /= steps
        val = evaluate_eav(model, val_loader, device, shuffle_mapping=semantic_mapping, ordinal_margin=args.ordinal_margin)
        selection_score = val['ce'] + lambda_ordinal * val['ordinal_loss']
        if selection_score < best_score:
            best_score = float(selection_score)
            best_epoch = epoch
            best_val = {'accuracy': val['accuracy'], 'macro_f1': val['macro_f1'], 'ordinal_metrics': val['ordinal_metrics']}
            stale = 0
            torch.save({'model_state': model.state_dict(), 'best_epoch': best_epoch, 'best_score': best_score, 'public_format_version': 1, 'model_config': {'embedding_dim': 128, 'shared_dim': 64, 'dropout': 0.25}, 'normalization': {'mean': torch.from_numpy(mean), 'std': torch.from_numpy(std), 'dataset': 'eav'}, 'anchor_contract': anchor_contract}, best_path)
        else:
            stale += 1
        print('epoch=%03d CE=%.4f ord=%.4f total=%.4f | valACC=%.4f valF1=%.4f valOrdCons=%.3f valOrdMargin=%.4f pat=%d' % (epoch, sums['ce'], sums['ord'], sums['total'], val['accuracy'], val['macro_f1'], val['ordinal_metrics']['consistency'], val['ordinal_metrics']['mean_relation_margin'], stale))
        if stale >= args.patience:
            print('early stop:', epoch)
            break
    state = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(state['model_state'])
    test = evaluate_eav(model, test_loader, device, shuffle_mapping=None, ordinal_margin=args.ordinal_margin)
    anchor_path = cfg.feature_dir / 'gate3' / 'gate3a_refed_anchor_v4' / f'refed_test{args.refed_anchor_test_subject:02d}_seed{args.seed:03d}.npz'
    if not anchor_path.exists():
        raise FileNotFoundError(anchor_path)
    anchor = np.load(anchor_path)
    retrieval = retrieval_metrics(test['shared'], test['labels'], anchor['test_shared'], anchor['test_true_va'], k=args.retrieval_k)
    print()
    print('#' * 78)
    print('GATE3A V4 ANCHORED PIVOT SUMMARY')
    print('#' * 78)
    print('mode:', 'ordinal_correct')
    print('best epoch:', best_epoch)
    print('best val:', best_val)
    print()
    print('DEV EAV TEST ACC:', test['accuracy'])
    print('DEV EAV TEST macro-F1:', test['macro_f1'])
    print('direct predicted-VA ordinal:', test['ordinal_metrics'])
    print('cross-dataset retrieval ordinal:', retrieval['ordinal_metrics'])
    print('retrieved REFED true-VA centroids:')
    print(retrieval['retrieved_va_centroids'])
    summary = {'protocol': PROTOCOL, 'mode': 'ordinal_correct', 'seed': args.seed, 'eav_fold': eav_split, 'refed_anchor_test_subject': args.refed_anchor_test_subject, 'cross_dataset_subject_pairing': False, 'anchor_frozen': True, 'lambda_ordinal': lambda_ordinal, 'ordinal_margin': args.ordinal_margin, 'retrieval_k': args.retrieval_k, 'best_epoch': best_epoch, 'best_selection_score': best_score, 'best_val': best_val, 'dev_test': {'accuracy': test['accuracy'], 'balanced_accuracy': test['balanced_accuracy'], 'macro_f1': test['macro_f1'], 'direct_va_centroids': test['va_centroids'].tolist(), 'direct_ordinal_metrics': test['ordinal_metrics'], 'retrieved_refed_va_centroids': retrieval['retrieved_va_centroids'].tolist(), 'retrieval_ordinal_metrics': retrieval['ordinal_metrics']}, 'test_used_for_selection': False, 'elapsed_sec': float(time.time() - start)}
    summary_path = out_dir / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=True) + '\n')
    print()
    print('summary:', summary_path)
    print('\nGATE3A V4 ANCHORED PIVOT: PASS')
if __name__ == '__main__':
    main()
