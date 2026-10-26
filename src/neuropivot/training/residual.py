from neuropivot.integrations.affectgpt.backend import validate_model
from neuropivot.paths import REPO, WORKDIR, OUTER_FOLD, OUTER_FOLDS, TRAIN_SUBJECTS_PUBLIC, N_TRAIN, N_TRIALS, fold_text
from pathlib import Path
import os
from neuropivot.models.residual import ResidualAdapter
import os
import sys
import gc
import json
import random
import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
ROOT = REPO
SCRIPTS = ROOT / 'scripts'
AFFECTGPT_ROOT = Path(os.environ.get('AFFECTGPT_ROOT', 'third_party/AffectGPT/AffectGPT'))
from neuropivot.data import av_physio_data as v4
stageb = v4.stageb
CORRECT_EMOTIONS = ['neutral', 'anger', 'happiness', 'sadness', 'calmness']
stageb.EMOTIONS = list(CORRECT_EMOTIONS)
TRAIN_SUBJECTS = TRAIN_SUBJECTS_PUBLIC
VALIDATION_SUBJECT = 2
OUTER_TEST_SUBJECTS = OUTER_FOLDS[OUTER_FOLD]
PERMANENTLY_EXCLUDED_SUBJECTS = [1]
AV_ROOT = WORKDIR / 'features' / 'gate4' / 'gate4_affectgpt_av_llm_embeddings_v1'
PHYS_ROOT = WORKDIR / 'features' / 'gate4' / fold_text('gate4_eav_grouped_outer01_pivot_v1')
TASK_CKPT = WORKDIR / 'checkpoints' / 'gate4' / fold_text('gate4_affectgpt_task_adapter_grouped_outer01_label_fix_v1') / 'seed042' / 'best.pt'
OUT_ROOT = WORKDIR / 'checkpoints' / 'gate4' / fold_text('gate4_affectgpt_pivot_residual_grouped_outer01_label_fix_v1')
ALL_CONDITIONS = ['va_pivot_real']

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg-path', default='train_configs/mercaptionplus_outputhybird_bestsetup_bestfusion_frame_lz.yaml')
    p.add_argument('--options', nargs='+', default=['inference.test_epoch=30', 'inference.gpu=0'])
    p.add_argument('--checkpoint', default='output/mercaptionplus_outputhybird_bestsetup_bestfusion_frame_lz/mercaptionplus_outputhybird_bestsetup_bestfusion_frame_lz_20250408110/checkpoint_000030_loss_0.751.pth')
    p.set_defaults(conditions=['va_pivot_real'])
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--grad-accum', type=int, default=8)
    p.add_argument('--lr', type=float, default=0.0001)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-steps-per-epoch', type=int, default=0)
    p.add_argument('--run-tag', default='')
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args()

def load_rows(subjects, representation, phys_root=PHYS_ROOT):
    fold_text('\n    Reuse the exact V4 cache interface.\n\n    This is important because the frozen Stage-B Builder expects\n    precisely the tensor dimensionality produced by the validated\n    V4 load_subject()/CachedDataset pipeline.\n\n    Grouped Outer01 cache. This trainer loads only TRAIN_SUBJECTS.\n    Outer-test subjects remain sealed until separate evaluation.\n    ')
    old_av_root = v4.AV_ROOT
    old_phys_root = v4.PHYS_ROOT
    try:
        v4.AV_ROOT = AV_ROOT
        v4.PHYS_ROOT = Path(phys_root)
        rows = v4.load_subjects(subjects, 'va_pivot')
    finally:
        v4.AV_ROOT = old_av_root
        v4.PHYS_ROOT = old_phys_root
    return rows

def prepare_train_features(rows, mode, seed, feature_dim):
    assert len(rows) == N_TRIALS
    subjects = sorted({int(r['subject']) for r in rows})
    assert subjects == TRAIN_SUBJECTS
    x = np.stack([r['feature'] for r in rows], axis=0).astype(np.float32)
    assert x.shape == (N_TRIALS, 66)
    mean = x.mean(axis=0).astype(np.float32)
    std = x.std(axis=0).astype(np.float32)
    if not np.isfinite(mean).all():
        raise RuntimeError('nonfinite feature mean')
    if not np.isfinite(std).all():
        raise RuntimeError('nonfinite feature std')
    if float(std.min()) <= 1e-06:
        raise RuntimeError(f'near-constant physiology dim: {float(std.min())}')
    for r in rows:
        r['feature_used'] = r['feature'].copy()
    pass
    for r in rows:
        z = ((r['feature_used'] - mean) / std).astype(np.float32)
        assert z.shape == (66,)
        assert np.isfinite(z).all()
        r['z'] = z
    return (mean, std)

class CachedDataset(Dataset):

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return {'label': torch.tensor(r['label'], dtype=torch.long), 'z': torch.from_numpy(r['z']).float(), 'audio': torch.from_numpy(r['audio']), 'frame': torch.from_numpy(r['frame']), 'multi': torch.from_numpy(r['multi'])}

def condition_spec(name):
    return ('va_pivot', 'real', 66)

def train_condition(args, model, device, task_token, condition):
    representation, mode, feature_dim = condition_spec('va_pivot_real')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print()
    print('#' * 88)
    print('FINAL CONDITION:', 'va_pivot_real')
    print('#' * 88)
    print('representation:', 'va_pivot')
    print('mode:', 'real')
    print('feature dim:', 66)
    rows = load_rows(TRAIN_SUBJECTS, 'va_pivot')
    assert len(rows) == N_TRIALS
    mean, std = prepare_train_features(rows, 'real', args.seed, 66)
    print()
    print('feature std min/median/max:', float(std.min()), float(np.median(std)), float(std.max()))
    ds = v4.CachedDataset(rows)
    g = torch.Generator()
    g.manual_seed(args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, generator=g, num_workers=0, pin_memory=True, drop_last=False)
    adapter = ResidualAdapter(66).to(device)
    expected = 467200
    n_trainable = sum((p.numel() for p in adapter.parameters() if p.requires_grad))
    assert n_trainable == expected
    with torch.no_grad():
        zero = torch.zeros(8, 66, device=device)
        delta = adapter(zero)
        assert torch.equal(delta, torch.zeros_like(delta))
    print('CENTERED ZERO -> EXACT ZERO RESIDUAL: PASS')
    builder = stageb.Builder(model, device, task_token, adapter)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=0.0)
    run_name = f"{'va_pivot_real'}_seed{args.seed:03d}"
    if args.run_tag:
        run_name += '_' + args.run_tag
    out_dir = OUT_ROOT / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / 'final.pt'
    metrics_path = out_dir / 'metrics.json'
    if final_path.is_file() and (not args.overwrite):
        print('already complete; skip:', final_path)
        return
    history = []
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, args.epochs + 1):
        print()
        print('=' * 80)
        print(f"{'va_pivot_real'} EPOCH {epoch}/{args.epochs}")
        print('=' * 80)
        adapter.train()
        model.llama_model.eval()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        ratio_sum = 0.0
        n = 0
        steps_total = len(loader)
        if args.max_steps_per_epoch > 0:
            steps_total = min(steps_total, args.max_steps_per_epoch)
        for step, batch in enumerate(loader, start=1):
            if step > steps_total:
                break
            loss, ratio = stageb.forward_loss(model, builder, batch, use_residual=True)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{'va_pivot_real'}: nonfinite loss")
            (loss / args.grad_accum).backward()
            bad = [name for name, p in model.named_parameters() if p.grad is not None]
            if bad:
                raise RuntimeError(f'Frozen AffectGPT received gradients: {bad[:10]}')
            if task_token.grad is not None:
                raise RuntimeError('Frozen task token received gradient')
            if step % args.grad_accum == 0 or step == steps_total:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            b = int(batch['label'].shape[0])
            raw = float(loss.detach().cpu())
            ratio_f = float(ratio)
            loss_sum += raw * b
            ratio_sum += ratio_f * b
            n += b
            if step == 1 or step % 100 == 0 or step == steps_total:
                free, _ = torch.cuda.mem_get_info(0)
                print(f'step={step:04d}/{steps_total:04d} loss={raw:.6f} avg={loss_sum / n:.6f} ratio={ratio_f:.6f} freeGB={free / 1024 ** 3:.2f}')
        train_loss = loss_sum / n
        train_ratio = ratio_sum / n
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        history.append({'epoch': epoch, 'train_loss': float(train_loss), 'train_residual_multi_ratio': float(train_ratio), 'peak_allocated_gb': float(peak), 'steps': int(steps_total)})
        print()
        print('train_loss:', train_loss)
        print('train residual/multi ratio:', train_ratio)
    scientific_full_run = args.max_steps_per_epoch == 0 and args.epochs == 3 and (args.batch_size == 2) and (args.grad_accum == 8) and (abs(args.lr - 0.0001) < 1e-15) and (args.seed == 42)
    checkpoint = {'adapter': {k: v.detach().cpu().float() for k, v in adapter.state_dict().items()}, 'epoch': args.epochs, 'representation': 'va_pivot', 'mode': 'real', 'condition': 'va_pivot_real', 'feature_dim': 66, 'shuffle_scope': 'none', 'feature_mean': torch.from_numpy(mean).float(), 'feature_std': torch.from_numpy(std).float(), 'seed': args.seed, 'task_checkpoint': str(TASK_CKPT), 'protocol': fold_text('gate4_grouped_outer01_residual_fit_label_fix_v1'), 'fit_role': 'fixed_epoch_final_fit', 'train_subjects': TRAIN_SUBJECTS, 'train_trials': N_TRIALS, 'validation_subject': VALIDATION_SUBJECT, 'validation_subject_used_for_fit': False, 'outer_test_subjects': OUTER_TEST_SUBJECTS, 'outer_test_subjects_used_for_fit': False, 'permanently_excluded_subjects': PERMANENTLY_EXCLUDED_SUBJECTS, 'subject01_used_for_fit': False, 'epochs': args.epochs, 'lr': args.lr, 'batch_size': args.batch_size, 'grad_accum': args.grad_accum, 'pivot_l2_normalized': False, 'scientific_full_run': scientific_full_run, 'anchor_contract': json.loads((WORKDIR / 'checkpoints/gate3/neuropivot_anchor_contract.json').read_text(encoding='utf-8'))}
    torch.save(checkpoint, final_path)
    metrics = {'status': 'PASS', 'protocol': fold_text('gate4_grouped_outer01_residual_fit_label_fix_v1'), 'condition': 'va_pivot_real', 'representation': 'va_pivot', 'mode': 'real', 'feature_dim': 66, 'shuffle_scope': 'none', 'train_subjects': TRAIN_SUBJECTS, 'train_trials': N_TRIALS, 'validation_subject': VALIDATION_SUBJECT, 'validation_subject_used_for_fit': False, 'outer_test_subjects': OUTER_TEST_SUBJECTS, 'outer_test_subjects_used_for_fit': False, 'permanently_excluded_subjects': PERMANENTLY_EXCLUDED_SUBJECTS, 'subject01_used_for_fit': False, 'task_checkpoint': str(TASK_CKPT), 'physiology_cache': str(PHYS_ROOT), 'feature_mean': mean.tolist(), 'feature_std': std.tolist(), 'trainable_params': n_trainable, 'epochs': args.epochs, 'lr': args.lr, 'batch_size': args.batch_size, 'grad_accum': args.grad_accum, 'scientific_full_run': scientific_full_run, 'history': history, 'checkpoint': str(final_path)}
    metrics_path.write_text(json.dumps(metrics, indent=2))
    reloaded = torch.load(final_path, map_location='cpu')
    assert reloaded['condition'] == 'va_pivot_real'
    assert reloaded['train_subjects'] == TRAIN_SUBJECTS
    assert reloaded['subject01_used_for_fit'] is False
    assert reloaded['validation_subject'] == VALIDATION_SUBJECT
    assert reloaded['validation_subject_used_for_fit'] is False
    assert reloaded['outer_test_subjects'] == OUTER_TEST_SUBJECTS
    assert reloaded['outer_test_subjects_used_for_fit'] is False
    if args.max_steps_per_epoch == 0:
        assert reloaded['scientific_full_run'] is True
    print()
    print('va_pivot_real', 'FINAL CHECKPOINT: PASS')
    print(final_path)
    del optimizer
    del builder
    del adapter
    del loader
    del ds
    del rows
    gc.collect()
    torch.cuda.empty_cache()

def main():
    args = parse_args()
    if args.max_steps_per_epoch > 0 and (not args.run_tag):
        raise RuntimeError('Smoke run requires --run-tag so it cannot be mistaken for final.')
    if args.max_steps_per_epoch == 0:
        assert args.epochs == 3
        assert args.batch_size == 2
        assert args.grad_accum == 8
        assert abs(args.lr - 0.0001) < 1e-15
        assert args.seed == 42
    assert len(TRAIN_SUBJECTS) == N_TRAIN
    assert len(OUTER_TEST_SUBJECTS) == len(OUTER_FOLDS[OUTER_FOLD])
    assert set(TRAIN_SUBJECTS).isdisjoint(OUTER_TEST_SUBJECTS)
    protocol_subjects = PERMANENTLY_EXCLUDED_SUBJECTS + [VALIDATION_SUBJECT] + TRAIN_SUBJECTS + OUTER_TEST_SUBJECTS
    assert sorted(protocol_subjects) == list(range(1, 43))
    assert len(set(protocol_subjects)) == 42
    assert not (PHYS_ROOT / 'subject01.npz').exists()
    assert not (PHYS_ROOT / 'subject02.npz').exists()
    for subject in TRAIN_SUBJECTS + OUTER_TEST_SUBJECTS:
        assert (PHYS_ROOT / f'subject{subject:02d}.npz').is_file()
    gc.collect()
    print('VA_PIVOT FIRST-2D VA ALIGNMENT: PASS')
    if not TASK_CKPT.is_file():
        raise RuntimeError(f'Missing frozen task token: {TASK_CKPT}')
    try:
        task_metadata = torch.load(TASK_CKPT, map_location='cpu', weights_only=True)
    except TypeError:
        task_metadata = torch.load(TASK_CKPT, map_location='cpu')
    assert task_metadata['train_subjects'] == TRAIN_SUBJECTS
    assert task_metadata['validation_subject'] == VALIDATION_SUBJECT
    assert task_metadata['outer_test_subjects'] == OUTER_TEST_SUBJECTS
    assert task_metadata['permanently_excluded_subjects'] == PERMANENTLY_EXCLUDED_SUBJECTS
    assert task_metadata['outer_test_trials_used_for_training'] == 0
    assert task_metadata['outer_test_trials_used_for_selection'] == 0
    print('GROUPED STAGE-A METADATA: PASS')
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == 1
    device = torch.device('cuda:0')
    print('=' * 88)
    print(fold_text('GATE4 GROUPED OUTER01 MAIN PHYSIOLOGY RESIDUAL FIT'))
    print('=' * 88)
    print('GPU:', torch.cuda.get_device_name(0))
    print('train subjects:', TRAIN_SUBJECTS)
    print('train trials:', N_TRIALS)
    print('validation subject:', VALIDATION_SUBJECT, '(not used for residual fit)')
    print('sealed outer test subjects:', OUTER_TEST_SUBJECTS)
    print('permanently excluded subjects:', PERMANENTLY_EXCLUDED_SUBJECTS)
    print('frozen Stage-A:', TASK_CKPT)
    print('conditions:', args.conditions)
    cfg = stageb.Config(args)
    model_cfg = cfg.model_cfg
    affect_ckpt = Path(args.checkpoint)
    if not affect_ckpt.is_file():
        raise RuntimeError(f'Missing AffectGPT checkpoint: {affect_ckpt}')
    model_cfg.ckpt_3 = str(affect_ckpt)
    model_cls = stageb.registry.get_model_class(model_cfg.arch)
    model = model_cls.from_config(model_cfg)
    validate_model(model)
    for p in model.parameters():
        p.requires_grad = False
    print()
    print('======== CACHED-AV MEMORY PRUNE ========')
    free_before, _ = torch.cuda.mem_get_info(0)
    print('before_prune_free_GB:', free_before / 1024 ** 3)
    retained_llama = model.llama_model
    removed = []
    for child_name, child_module in list(model.named_children()):
        if child_name != 'llama_model':
            removed.append(child_name)
            setattr(model, child_name, None)
    del child_module
    model.llama_model = retained_llama
    del retained_llama
    gc.collect()
    torch.cuda.empty_cache()
    llama_dtype = next(model.llama_model.parameters()).dtype
    if llama_dtype == torch.float32:
        model.llama_model = model.llama_model.half()
        gc.collect()
        torch.cuda.empty_cache()
    llama_dtype = next(model.llama_model.parameters()).dtype
    assert llama_dtype in {torch.float16, torch.bfloat16}
    model.llama_model = model.llama_model.to(device).eval()
    gc.collect()
    torch.cuda.empty_cache()
    free_after, _ = torch.cuda.mem_get_info(0)
    print('removed unused child modules:', removed)
    print('llama dtype:', llama_dtype)
    print('after_llm_move_free_GB:', free_after / 1024 ** 3)
    print('CACHED_AV_PRUNE: PASS')
    model.llama_model.config.use_cache = False
    assert not any((p.requires_grad for p in model.parameters()))
    task_obj = task_metadata
    assert tuple(task_obj['task_token'].shape) == (1, 1, 3584)
    task_token = task_obj['task_token'].float().to(device)
    task_token.requires_grad_(False)
    print('Stage-A checkpoint epoch:', task_obj['epoch'])
    print('Stage-A frozen dev02 NLL:', task_obj['val_loss'])
    print('AffectGPT trainable: 0')
    print('task token trainable: 0')
    for condition in args.conditions:
        train_condition(args, model, device, task_token, 'va_pivot_real')
    print()
    print('=' * 88)
    if args.max_steps_per_epoch == 0:
        print(fold_text('ALL GROUPED OUTER01 CONDITIONS: FROZEN / PASS'))
    else:
        print('FINAL TRAINER ENGINEERING SMOKE: PASS')
        print('DO NOT INTERPRET SCIENTIFICALLY')
    print('=' * 88)
if __name__ == '__main__':
    main()
