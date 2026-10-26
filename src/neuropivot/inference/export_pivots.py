from neuropivot.training.contracts import bind_anchor, assert_checkpoint_anchor
from neuropivot.paths import REPO, WORKDIR, OUTER_FOLD, OUTER_FOLDS, TRAIN_SUBJECTS_PUBLIC, N_TRAIN, N_TRIALS, fold_text
from pathlib import Path
import os
import json
FOLD_SPEC = json.loads((REPO / 'configs' / 'folds.json').read_text(encoding='utf-8'))[str(OUTER_FOLD)]
ROLE_TESTS = FOLD_SPEC['ROLE_TESTS']
ROLE_JOB_IDS = FOLD_SPEC['ROLE_JOB_IDS']
import argparse
import gc
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
ROOT = REPO
UPSTREAM_PROTOCOL = 'gate3a_anchored_pivot_v4_grouped_outer5fold_v1'
CACHE_PROTOCOL = fold_text('gate4_eav_grouped_outer01_pivot_v1')
OLD_METADATA_PROTOCOL = 'eav_trial_metadata_v1'
SEED = 42
PIVOT_DIM = 64
OUTER_TEST = OUTER_FOLDS[OUTER_FOLD]
TRAIN_SIDE = sorted(set(range(3, 43)) - set(OUTER_TEST))
PERMANENTLY_FORBIDDEN = sorted({1, *OUTER_TEST})
ROLE_ORDER = ['inner1', 'inner2', 'inner3', 'inner4', 'inner5', 'outer']

def parse_subjects(text):
    out = []
    for part in text.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = (int(x) for x in part.split('-', 1))
            if b < a:
                raise ValueError(f'bad range: {part}')
            out.extend(range(a, b + 1))
        else:
            out.append(int(part))
    out = sorted(set(out))
    if not out:
        raise ValueError('empty subject list')
    bad = sorted(set(out) - set(range(3, 43)))
    if bad:
        raise ValueError(f'only EAV subjects03-42 are permitted; got {bad}')
    return out

def build_specs():
    specs = {}
    for role in ROLE_ORDER:
        test = list(ROLE_TESTS[role])
        train = TRAIN_SIDE if role == 'outer' else sorted(set(TRAIN_SIDE) - set(test))
        specs[role] = {'fold_id': f"{fold_text('grouped_outer01_')}{role}", 'outer_fold': OUTER_FOLD, 'split_role': role, 'inner_fold': None if role == 'outer' else int(role[-1]), 'outer_test_subjects': list(OUTER_TEST), 'permanently_forbidden_from_train_val': list(PERMANENTLY_FORBIDDEN), 'train': list(train), 'val': [2], 'test': test, 'job_id': ROLE_JOB_IDS[role]}
    return specs

def verify_specs(specs):
    inner_sets = [set(specs[f'inner{i}']['test']) for i in range(1, 6)]
    if set().union(*inner_sets) != set(TRAIN_SIDE):
        raise RuntimeError('inner test folds do not cover the complete training side')
    for i, left in enumerate(inner_sets):
        for right in inner_sets[i + 1:]:
            if left & right:
                raise RuntimeError('inner test folds overlap')
    if set(specs['outer']['test']) != set(OUTER_TEST):
        raise RuntimeError('outer test split mismatch')
    if set(specs['outer']['train']) != set(TRAIN_SIDE):
        raise RuntimeError('outer train split mismatch')
    for role, split in specs.items():
        train, val, test = map(set, (split['train'], split['val'], split['test']))
        if val != {2}:
            raise RuntimeError(f'{role}: validation subject must be 02')
        if train & val or train & test or val & test:
            raise RuntimeError(f'{role}: split overlap')
        if set(PERMANENTLY_FORBIDDEN) & (train | val):
            raise RuntimeError(f'{role}: outer leakage into train/val')
        if 1 in train | val | test:
            raise RuntimeError(f'{role}: subject01 must remain excluded')

def role_for_subject(subject, specs):
    roles = [role for role, split in specs.items() if subject in split['test']]
    if len(roles) != 1:
        raise RuntimeError(f'subject{subject:02d}: expected one cache role, got {roles}')
    return roles[0]

def checkpoint_path(cfg, job_id):
    return cfg.checkpoint_dir / 'gate3' / UPSTREAM_PROTOCOL / f'eav_test{job_id:02d}_ordinal_correct_seed042_tmm_grouped_v1' / 'best.pt'

def assert_static_inputs(cfg, subjects, specs, base):
    for role in ROLE_ORDER:
        path = checkpoint_path(cfg, specs[role]['job_id'])
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata_root = cfg.feature_dir / 'gate4' / OLD_METADATA_PROTOCOL
    window_root = cfg.feature_dir / 'gate3' / base.v4.v2.EAV_CACHE_PROTOCOL
    for subject in subjects:
        for path in (metadata_root / f'subject{subject:02d}.npz', window_root / f'subject{subject:02d}.npz'):
            if not path.is_file():
                raise FileNotFoundError(path)

def aligned_metadata(metadata_path, trial, subject):
    metadata = np.load(metadata_path, allow_pickle=True)
    required = ['sequence_id', 'parent_trial_group_id', 'label', 'audio_path', 'video_path']
    missing = [key for key in required if key not in metadata.files]
    if missing:
        raise RuntimeError(f'{metadata_path}: missing metadata keys {missing}')
    sequence_id = metadata['sequence_id'].astype(np.int64)
    parent = metadata['parent_trial_group_id'].astype(str)
    label = metadata['label'].astype(np.int64)
    audio_path = metadata['audio_path'].astype(str)
    video_path = metadata['video_path'].astype(str)
    if not np.array_equal(trial['sequence_id'], sequence_id):
        raise RuntimeError(f'subject{subject:02d}: sequence mismatch')
    if not np.array_equal(trial['parent_trial_group_id'].astype(str), parent):
        raise RuntimeError(f'subject{subject:02d}: parent-trial mismatch')
    if not np.array_equal(trial['label'], label):
        raise RuntimeError(f'subject{subject:02d}: label mismatch')
    if audio_path.shape != (100,) or video_path.shape != (100,):
        raise RuntimeError(f'subject{subject:02d}: bad audio/video metadata shape')
    return (required, audio_path, video_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths-config', type=Path, required=True)
    parser.add_argument('--refed-anchor-test-subject',type=int,choices=[1,2],required=True)
    parser.add_argument('--subjects', default='3-42')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    if args.seed != SEED:
        raise RuntimeError(fold_text('Outer01 grouped cache protocol is frozen to seed42'))
    from neuropivot.inference import pivot_cache_base as base
    specs = build_specs()
    verify_specs(specs)
    subjects = parse_subjects(args.subjects)
    cfg = base.load_config(args.paths_config)
    anchor_contract = bind_anchor(cfg, args.refed_anchor_test_subject, args.seed)
    assert_static_inputs(cfg, subjects, specs, base)
    print('=' * 80)
    print(fold_text('GATE4 GROUPED OUTER01 VA + 64D PIVOT CACHE V1'))
    print('=' * 80)
    print('subjects:', subjects)
    print('upstream protocol:', UPSTREAM_PROTOCOL)
    print('outer test:', OUTER_TEST)
    print('permanently forbidden from every upstream train/val:', PERMANENTLY_FORBIDDEN)
    for role in ROLE_ORDER:
        selected = [s for s in subjects if role_for_subject(s, specs) == role]
        print(f"{role}: job={specs[role]['job_id']:02d} cache_subjects={selected}")
    if args.audit_only:
        print(fold_text('GATE4 GROUPED OUTER01 CACHE PREFLIGHT: PASS'))
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('expected exactly one visible GPU')
    device = torch.device('cuda:0')
    print('GPU:', torch.cuda.get_device_name(0))
    out_root = cfg.feature_dir / 'gate4' / CACHE_PROTOCOL
    metadata_root = cfg.feature_dir / 'gate4' / OLD_METADATA_PROTOCOL
    out_root.mkdir(parents=True, exist_ok=True)
    records = []
    for role in ROLE_ORDER:
        split = specs[role]
        selected = [s for s in subjects if role_for_subject(s, specs) == role]
        pending = [s for s in selected if args.overwrite or not (out_root / f'subject{s:02d}.npz').is_file()]
        if not pending:
            print(f'{role}: all selected cache files already exist; skip')
            continue
        print('=' * 80)
        print(f"ROLE {role} | JOB {split['job_id']:02d} | SUBJECTS {pending}")
        print('train:', split['train'])
        print('val:', split['val'])
        print('test:', split['test'])
        train_x, _ = base.v4.load_eav_group(cfg, split['train'])
        mean, std = base.v4.v2.channel_stats(train_x)
        del train_x
        if mean.shape != (1, 30, 1) or std.shape != (1, 30, 1):
            raise RuntimeError(f'{role}: bad channel-stat shapes {mean.shape}, {std.shape}')
        ckpt_path = checkpoint_path(cfg, split['job_id'])
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        assert_checkpoint_anchor(state,args.refed_anchor_test_subject)
        model = base.Gate3EEGPivotV3().to(device)
        model.load_state_dict(state['model_state'])
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        for subject in pending:
            print(f"building subject{subject:02d} with {role}/job{split['job_id']:02d} ...")
            cache = base.load_window_cache(cfg, subject)
            window_va, window_shared = base.predict_windows(model, cache, mean, std, args.batch_size, device)
            trial = base.aggregate_trials(cache, window_va, window_shared)
            metadata_path = metadata_root / f'subject{subject:02d}.npz'
            metadata_required, audio_path, video_path = aligned_metadata(metadata_path, trial, subject)
            va = trial['teacher_va']
            shared = trial['pivot_shared']
            if not np.isfinite(va).all() or not np.isfinite(shared).all():
                raise RuntimeError(f'subject{subject:02d}: nonfinite representation')
            norms = np.linalg.norm(shared, axis=1)
            n_unique = int(np.unique(shared, axis=0).shape[0])
            if n_unique < 95:
                raise RuntimeError(f'subject{subject:02d}: pivot collapsed, unique={n_unique}')
            output_path = out_root / f'subject{subject:02d}.npz'
            np.savez_compressed(output_path, subject_number=np.full(100, subject, dtype=np.int64), sequence_id=trial['sequence_id'], parent_trial_group_id=trial['parent_trial_group_id'].astype(str), label=trial['label'], teacher_va=va.astype(np.float32), teacher_window_std=trial['teacher_window_std'].astype(np.float32), pivot_shared=shared.astype(np.float32), pivot_window_std=trial['pivot_window_std'].astype(np.float32), audio_path=audio_path, video_path=video_path, teacher_eav_test_subject=np.full(100, subject, dtype=np.int64), teacher_grouped_job_id=np.asarray(split['job_id'], dtype=np.int64), teacher_checkpoint=np.asarray([str(ckpt_path)] * 100, dtype=np.str_), upstream_protocol=np.asarray(UPSTREAM_PROTOCOL), upstream_seed=np.asarray(SEED, dtype=np.int64), upstream_refed_anchor_test_subject=np.asarray(args.refed_anchor_test_subject, dtype=np.int64), upstream_fold_id=np.asarray(split['fold_id']), upstream_outer_fold=np.asarray(OUTER_FOLD, dtype=np.int64), upstream_split_role=np.asarray(role), upstream_inner_fold=np.asarray(-1 if split['inner_fold'] is None else split['inner_fold'], dtype=np.int64), outer_test_subjects=np.asarray(OUTER_TEST, dtype=np.int64), permanently_forbidden_from_upstream_train_val=np.asarray(PERMANENTLY_FORBIDDEN, dtype=np.int64), nested_train_subjects=np.asarray(split['train'], dtype=np.int64), nested_val_subjects=np.asarray(split['val'], dtype=np.int64), nested_test_subjects=np.asarray(split['test'], dtype=np.int64), pivot_dim=np.asarray(PIVOT_DIM, dtype=np.int64), pivot_l2_normalized=np.asarray(False, dtype=np.bool_), metadata_source=np.asarray(str(metadata_path)), metadata_fields_used=np.asarray(metadata_required))
            print('saved:', output_path, 'VA_mean:', va.mean(axis=0).tolist(), 'pivot_norm_mean:', float(norms.mean()), 'unique:', n_unique)
            records.append({'subject': subject, 'role': role, 'job_id': split['job_id'], 'train_subjects': split['train'], 'val_subjects': split['val'], 'test_subjects': split['test'], 'checkpoint': str(ckpt_path), 'path': str(output_path), 'va_mean': va.mean(axis=0).tolist(), 'va_std': va.std(axis=0).tolist(), 'pivot_norm_mean': float(norms.mean()), 'pivot_norm_std': float(norms.std()), 'pivot_unique_trials': n_unique})
        del model, state, mean, std
        gc.collect()
        torch.cuda.empty_cache()
    audit = {'status': 'PASS', 'protocol': CACHE_PROTOCOL, 'created_utc': datetime.now(timezone.utc).isoformat(), 'subjects_requested': subjects, 'subjects_built_this_run': [record['subject'] for record in records], 'outer_fold': OUTER_FOLD, 'outer_test_subjects': OUTER_TEST, 'permanently_forbidden_from_upstream_train_val': PERMANENTLY_FORBIDDEN, 'upstream_protocol': UPSTREAM_PROTOCOL, 'upstream_mode': 'ordinal_correct', 'upstream_seed': SEED, 'refed_anchor_test_subject': args.refed_anchor_test_subject, 'pivot_dim': PIVOT_DIM, 'posthoc_l2_normalization': False, 'old_oof_teacher_values_used': False, 'records': records}
    audit_path = cfg.audit_dir / 'gate4' / f'{CACHE_PROTOCOL}_audit.json'
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2))
    expected_paths = [out_root / f'subject{s:02d}.npz' for s in subjects]
    missing_outputs = [str(path) for path in expected_paths if not path.is_file()]
    if missing_outputs:
        raise RuntimeError(f'missing output caches: {missing_outputs}')
    print('=' * 80)
    print(fold_text('GATE4 GROUPED OUTER01 PIVOT CACHE BUILD: PASS'))
    print('cache files present:', len(expected_paths))
    print('built this run:', len(records))
    print('output:', out_root)
    print('audit:', audit_path)
if __name__ == '__main__':
    main()
