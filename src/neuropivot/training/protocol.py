from pathlib import Path
import os
REPO = Path(__file__).resolve().parents[3]
WORKDIR = Path(os.environ.get("NEUROPIVOT_WORKDIR", str(REPO / "work"))).resolve()
OUTER_FOLD = int(os.environ.get("NEUROPIVOT_OUTER_FOLD", "1"))

def fold_text(value):
    return value.replace("outer01", f"outer{OUTER_FOLD:02d}").replace("OUTER01", f"OUTER{OUTER_FOLD:02d}").replace("Outer01", f"Outer{OUTER_FOLD:02d}")

import sys
import json
import argparse
import importlib.util
from pathlib import Path
import numpy as np
ROOT = REPO
if not (ROOT / 'scripts').is_dir():
    ROOT = REPO
BASE_PATH = ROOT / 'scripts' / 'bridge_base.py'
PROTOCOL = 'gate3a_anchored_pivot_v4_grouped_outer5fold_v1'
RUN_TAG = 'tmm_grouped_v1'
SEED = 42
FOLD_SEED = 42005
OUTER_FOLDS = {1: [14, 18, 21, 23, 30, 32, 35, 41], 2: [5, 8, 13, 25, 26, 29, 31, 34], 3: [6, 12, 20, 33, 36, 37, 38, 40], 4: [4, 7, 15, 19, 22, 27, 39, 42], 5: [9, 10, 11, 16, 17, 24, 28]}
ROLE_TO_INDEX = {'outer': 0, 'inner1': 1, 'inner2': 2, 'inner3': 3, 'inner4': 4, 'inner5': 5}

def make_outer_folds():
    eligible = np.arange(4, 43, dtype=np.int64)
    rng = np.random.default_rng(FOLD_SEED)
    rng.shuffle(eligible)
    generated = {index + 1: sorted((int(x) for x in fold)) for index, fold in enumerate(np.array_split(eligible, 5))}
    if generated != OUTER_FOLDS:
        raise RuntimeError(f'Frozen outer folds do not reproduce from seed {FOLD_SEED}: {generated}')
    return generated

def inner_folds_for_outer(outer_fold):
    outer_test = set(OUTER_FOLDS[outer_fold])
    training_side = np.asarray([3, *[subject for subject in range(4, 43) if subject not in outer_test]], dtype=np.int64)
    rng = np.random.default_rng(42100 + outer_fold)
    rng.shuffle(training_side)
    inner = {index + 1: sorted((int(x) for x in fold)) for index, fold in enumerate(np.array_split(training_side, 5))}
    return inner

def build_split(outer_fold, role):
    if outer_fold not in OUTER_FOLDS:
        raise ValueError(f'outer_fold must be 1..5; got {outer_fold}')
    if role not in ROLE_TO_INDEX:
        raise ValueError(f'unknown role: {role}')
    outer_test = list(OUTER_FOLDS[outer_fold])
    outer_test_set = set(outer_test)
    training_side = sorted([3, *[subject for subject in range(4, 43) if subject not in outer_test_set]])
    inner = inner_folds_for_outer(outer_fold)
    if role == 'outer':
        train = list(training_side)
        test = list(outer_test)
        inner_fold = None
    else:
        inner_fold = int(role.replace('inner', ''))
        test = list(inner[inner_fold])
        test_set = set(test)
        train = [subject for subject in training_side if subject not in test_set]
    val = [2]
    permanently_forbidden = sorted({1, *outer_test})
    split = {'fold_id': f'grouped_outer{outer_fold:02d}_{role}', 'protocol': PROTOCOL, 'outer_fold': int(outer_fold), 'split_role': role, 'inner_fold': inner_fold, 'outer_test_subjects': outer_test, 'permanently_forbidden_from_train_val': permanently_forbidden, 'train': train, 'val': val, 'test': test}
    validate_split(split)
    return split

def validate_split(split):
    train = set(split['train'])
    val = set(split['val'])
    test = set(split['test'])
    outer_test = set(split['outer_test_subjects'])
    forbidden = set(split['permanently_forbidden_from_train_val'])
    if val != {2}:
        raise RuntimeError('Upstream validation must be subject02')
    if train & val:
        raise RuntimeError('train/val overlap')
    if train & test:
        raise RuntimeError('train/test overlap')
    if val & test:
        raise RuntimeError('val/test overlap')
    if train & forbidden:
        raise RuntimeError('permanently forbidden subject entered train')
    if val & forbidden:
        raise RuntimeError('permanently forbidden subject entered val')
    if 1 in train | val | test:
        raise RuntimeError('subject01 entered grouped-outer protocol')
    if split['split_role'] == 'outer':
        if test != outer_test:
            raise RuntimeError('outer test set mismatch')
        expected_train = {3} | set(range(4, 43)) - outer_test
        if train != expected_train:
            raise RuntimeError('outer train set mismatch')
    else:
        if test & outer_test:
            raise RuntimeError('outer-test subject entered inner test')
        if not test:
            raise RuntimeError('empty inner test fold')

def validate_protocol():
    folds = make_outer_folds()
    flattened = [subject for fold in folds.values() for subject in fold]
    if len(flattened) != 39:
        raise RuntimeError('outer folds do not contain 39 subjects')
    if len(set(flattened)) != 39:
        raise RuntimeError('outer folds overlap')
    if set(flattened) != set(range(4, 43)):
        raise RuntimeError('outer-fold coverage is not subjects04-42')
    all_splits = {}
    for outer_fold in range(1, 6):
        inner = inner_folds_for_outer(outer_fold)
        training_side = {3} | set(range(4, 43)) - set(OUTER_FOLDS[outer_fold])
        inner_flat = [subject for fold in inner.values() for subject in fold]
        if len(inner_flat) != len(training_side):
            raise RuntimeError(f'outer{outer_fold}: inner size mismatch')
        if set(inner_flat) != training_side:
            raise RuntimeError(f'outer{outer_fold}: inner coverage mismatch')
        if len(inner_flat) != len(set(inner_flat)):
            raise RuntimeError(f'outer{outer_fold}: inner folds overlap')
        all_splits[f'outer{outer_fold:02d}'] = {role: build_split(outer_fold, role) for role in ROLE_TO_INDEX}
    return all_splits

def job_id_for(outer_fold, role):
    return outer_fold * 10 + ROLE_TO_INDEX[role]

def print_preflight(all_splits, outer_fold, role):
    print('=' * 92)
    print('P0-05 GROUPED-OUTER GATE3 PREFLIGHT')
    print('=' * 92)
    print('protocol:', PROTOCOL)
    print('seed:', SEED)
    print('outer fold seed:', FOLD_SEED)
    print('subject01: permanently excluded')
    print('subject02: fixed upstream validation')
    print('subject03: training-side cross-fitted subject')
    print('outer evaluation subjects: 04..42')
    for fold in range(1, 6):
        print()
        print(f'OUTER {fold}: {OUTER_FOLDS[fold]}')
        inner = inner_folds_for_outer(fold)
        for inner_index in range(1, 6):
            print(f'  INNER {inner_index}: {inner[inner_index]}')
    selected = all_splits[f'outer{outer_fold:02d}'][role]
    print()
    print('SELECTED JOB')
    print('job id:', job_id_for(outer_fold, role))
    print(json.dumps(selected, indent=2))
    print()
    print('outer coverage: 39/39 PASS')
    print('inner cross-fit coverage: PASS')
    print('subject01 exclusion: PASS')
    print('subject02 validation role: PASS')
    print('outer-test leakage guards: PASS')
    print('P0-05 GROUPED-OUTER PREFLIGHT: PASS')
    print('No model was loaded.')
    print('No training was started.')
    print('No checkpoint was modified.')

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--outer-fold', type=int, choices=range(1, 6), required=True)
    parser.add_argument('--split-role', choices=list(ROLE_TO_INDEX), required=True)
    parser.add_argument('--preflight-only', action='store_true')
    known, remaining = parser.parse_known_args()
    all_splits = validate_protocol()
    if known.preflight_only:
        print_preflight(all_splits, known.outer_fold, known.split_role)
        return
    if '--eav-test-subject' in remaining:
        raise RuntimeError('Do not pass --eav-test-subject; it is assigned by the grouped protocol.')
    if '--run-tag' in remaining:
        raise RuntimeError('Do not pass --run-tag; it is frozen by the grouped protocol.')

    def cli_value(flag):
        if flag not in remaining:
            return None
        index = remaining.index(flag)
        if index + 1 >= len(remaining):
            raise RuntimeError(f'Missing value for {flag}')
        return remaining[index + 1]
    if cli_value('--mode') != 'ordinal_correct':
        raise RuntimeError('Grouped-outer protocol requires --mode ordinal_correct')
    seed_text = cli_value('--seed')
    if seed_text is not None and int(seed_text) != SEED:
        raise RuntimeError('Grouped-outer protocol is frozen to seed42')
    split = build_split(known.outer_fold, known.split_role)
    job_id = job_id_for(known.outer_fold, known.split_role)
    base = load_base()
    base.PROTOCOL = PROTOCOL

    def explicit_split(cfg, received_job_id):
        _ = cfg
        if int(received_job_id) != job_id:
            raise RuntimeError('Grouped job id mismatch')
        return dict(split)
    base.read_eav_loso = explicit_split
    sys.argv = [sys.argv[0], *remaining, '--eav-test-subject', str(job_id), '--run-tag', RUN_TAG]
    base.main()

def load_base():
    from neuropivot.training import bridge_base
    return bridge_base
