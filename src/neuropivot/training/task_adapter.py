from neuropivot.integrations.affectgpt.backend import validate_model
from neuropivot.paths import REPO, WORKDIR, OUTER_FOLD, OUTER_FOLDS, TRAIN_SUBJECTS_PUBLIC, N_TRAIN, N_TRIALS, fold_text
from pathlib import Path
import os
from neuropivot.integrations.affectgpt.backend import Config, registry
import os
import sys
import json
import random
import argparse
import gc
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
ROOT = REPO
AFFECTGPT_ROOT = Path(os.environ.get('AFFECTGPT_ROOT', 'third_party/AffectGPT/AffectGPT'))
EMOTIONS = ['neutral', 'anger', 'happiness', 'sadness', 'calmness']
CACHE_ROOT = WORKDIR / 'features' / 'gate4' / 'gate4_affectgpt_av_llm_embeddings_v1'

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg-path', default='train_configs/mercaptionplus_outputhybird_bestsetup_bestfusion_frame_lz.yaml')
    p.add_argument('--options', nargs='+', default=['inference.test_epoch=30', 'inference.gpu=0'])
    p.add_argument('--checkpoint', default='output/mercaptionplus_outputhybird_bestsetup_bestfusion_frame_lz/mercaptionplus_outputhybird_bestsetup_bestfusion_frame_lz_20250408110/checkpoint_000030_loss_0.751.pth')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--grad-accum', type=int, default=8)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()

def load_subjects(subjects):
    rows = []
    for s in subjects:
        p = CACHE_ROOT / f'subject{s:02d}.npz'
        assert p.is_file(), p
        z = np.load(p, allow_pickle=True)
        subject = z['subject_number'].astype(np.int64)
        sequence = z['sequence_id'].astype(np.int64)
        label = z['label'].astype(np.int64)
        audio = z['audio_llms'].astype(np.float16)
        frame = z['frame_llms'].astype(np.float16)
        multi = z['multi_llms'].astype(np.float16)
        assert len(label) == 100
        assert np.all(subject == s)
        assert set(label.tolist()) == {0, 1, 2, 3, 4}
        assert audio.shape == (100, 1, 3584)
        assert frame.shape == (100, 1, 3584)
        assert multi.shape == (100, 1, 3584)
        for i in range(100):
            rows.append({'subject': int(subject[i]), 'sequence_id': int(sequence[i]), 'label': int(label[i]), 'audio': audio[i].copy(), 'frame': frame[i].copy(), 'multi': multi[i].copy()})
    return rows

class CachedAVDataset(Dataset):

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return {'subject': r['subject'], 'sequence_id': r['sequence_id'], 'label': r['label'], 'audio': torch.from_numpy(r['audio']), 'frame': torch.from_numpy(r['frame']), 'multi': torch.from_numpy(r['multi'])}

class TaskPromptBuilder:
    """
    Stage-A builder.

    IMPORTANT:
    <PhysioHere> is used ONLY as an already-registered
    placeholder position for the generic task token.

    No physiology VA is loaded or used anywhere.
    """

    def __init__(self, model, device, task_token):
        self.model = model
        self.device = device
        self.task_token = task_token
        self.tok = model.llama_tokenizer
        self.prompt = '###Human: <Multi><MultiHere></Multi> <Audio><AudioHere></Audio> <Video><FrameHere></Video> <Task><PhysioHere></Task> Classify the emotion as neutral, sadness, anger, happiness, or calmness. Answer with exactly one emotion word. ###Assistant: '
        self.prompt_ids = self.tok(self.prompt, add_special_tokens=False)['input_ids']
        ids = torch.tensor(self.prompt_ids)
        checks = {'MULTI': model.MULTI_PATCH_TOKEN_ID, 'AUDIO': model.AUDIO_PATCH_TOKEN_ID, 'FRAME': model.FRAME_PATCH_TOKEN_ID, 'TASK_SLOT': model.PHYSIO_PATCH_TOKEN_ID}
        print()
        print('======== TASK PROMPT TOKENS ========')
        for name, token_id in checks.items():
            n = int((ids == token_id).sum().item())
            print(name, token_id, 'count=', n)
            assert n == 1

    def build(self, batch):
        ys = batch['label'].cpu().numpy().astype(int)
        all_ids = []
        all_labels = []
        for y in ys:
            target = EMOTIONS[y]
            target_ids = self.tok(target, add_special_tokens=False)['input_ids']
            if self.tok.eos_token_id is not None:
                target_ids = target_ids + [self.tok.eos_token_id]
            ids = self.prompt_ids + target_ids
            labels = [-100] * len(self.prompt_ids) + target_ids
            all_ids.append(ids)
            all_labels.append(labels)
        max_len = max((len(x) for x in all_ids))
        pad_id = self.tok.pad_token_id
        if pad_id is None:
            pad_id = self.tok.eos_token_id
        padded_ids = []
        padded_labels = []
        masks = []
        for ids, labels in zip(all_ids, all_labels):
            n_pad = max_len - len(ids)
            padded_ids.append(ids + [pad_id] * n_pad)
            padded_labels.append(labels + [-100] * n_pad)
            masks.append([1] * len(ids) + [0] * n_pad)
        input_ids = torch.tensor(padded_ids, dtype=torch.long, device=self.device)
        labels = torch.tensor(padded_labels, dtype=torch.long, device=self.device)
        attention = torch.tensor(masks, dtype=torch.long, device=self.device)
        temp = input_ids.clone()
        replace_ids = [self.model.MULTI_PATCH_TOKEN_ID, self.model.AUDIO_PATCH_TOKEN_ID, self.model.FRAME_PATCH_TOKEN_ID, self.model.PHYSIO_PATCH_TOKEN_ID]
        for token_id in replace_ids:
            temp[temp == token_id] = 0
        with torch.no_grad():
            base = self.model.llama_model.model.model.embed_tokens(temp)
        dtype = base.dtype
        audio = batch['audio'].to(self.device, dtype=dtype, non_blocking=True)
        frame = batch['frame'].to(self.device, dtype=dtype, non_blocking=True)
        multi = batch['multi'].to(self.device, dtype=dtype, non_blocking=True)
        B = input_ids.shape[0]
        task = self.task_token.to(dtype=dtype).expand(B, -1, -1)
        replacements = [(self.model.MULTI_PATCH_TOKEN_ID, multi), (self.model.AUDIO_PATCH_TOKEN_ID, audio), (self.model.FRAME_PATCH_TOKEN_ID, frame), (self.model.PHYSIO_PATCH_TOKEN_ID, task)]
        rows = []
        for b in range(B):
            cur = base[b]
            for token_id, feat in replacements:
                idx = torch.where(input_ids[b] == token_id)[0]
                assert idx.numel() == 1
                j = int(idx[0].item())
                cur = torch.cat([cur[:j], feat[b], cur[j + 1:]], dim=0)
            rows.append(cur)
        embeds = torch.stack(rows, dim=0)
        return (embeds, labels, attention)

def forward_loss(model, builder, batch):
    embeds, labels, attention = builder.build(batch)
    out = model.llama_model(inputs_embeds=embeds, attention_mask=attention, labels=labels, return_dict=True, use_cache=False)
    return out.loss

@torch.no_grad()
def evaluate(model, builder, loader):
    model.llama_model.eval()
    total = 0.0
    n = 0
    for batch in loader:
        loss = forward_loss(model, builder, batch)
        b = int(batch['label'].shape[0])
        total += float(loss.detach().cpu()) * b
        n += b
    return total / n

def main():
    args = parse_args()
    train_subjects_v4 = TRAIN_SUBJECTS_PUBLIC
    development_subject_v4 = 2
    outer_test_subjects = OUTER_FOLDS[OUTER_FOLD]
    permanently_excluded_subjects = [1]
    assert len(train_subjects_v4) == N_TRAIN
    assert len(outer_test_subjects) == len(OUTER_FOLDS[OUTER_FOLD])
    assert development_subject_v4 == 2
    assert permanently_excluded_subjects == [1]
    assert set(train_subjects_v4).isdisjoint(outer_test_subjects)
    assert development_subject_v4 not in train_subjects_v4
    assert development_subject_v4 not in outer_test_subjects
    all_protocol_subjects = permanently_excluded_subjects + [development_subject_v4] + train_subjects_v4 + outer_test_subjects
    assert sorted(all_protocol_subjects) == list(range(1, 43))
    assert len(set(all_protocol_subjects)) == 42
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == 1
    device = torch.device('cuda:0')
    print('=' * 80)
    print(fold_text('GATE4 GROUPED OUTER01 STAGE-A GENERIC TASK ADAPTER'))
    print('=' * 80)
    print('GPU:', torch.cuda.get_device_name(0))
    print('train subjects:', train_subjects_v4)
    print('development validation:', f'subject{development_subject_v4:02d}')
    print('sealed outer test subjects:', outer_test_subjects)
    print('permanently excluded subjects:', permanently_excluded_subjects)
    print('PHYSIOLOGY INPUT: NONE')
    print('EEG/VA INPUT: NONE')
    train_rows = load_subjects(train_subjects_v4)
    val_rows = load_subjects([development_subject_v4])
    assert len(train_rows) == N_TRIALS
    assert len(val_rows) == 100
    assert sorted({int(r['subject']) for r in train_rows}) == train_subjects_v4
    assert {int(r['subject']) for r in val_rows} == {development_subject_v4}
    assert not {int(r['subject']) for r in train_rows} & set(outer_test_subjects)
    train_ds = CachedAVDataset(train_rows)
    val_ds = CachedAVDataset(val_rows)
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=g, num_workers=0, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
    cfg = Config(args)
    model_cfg = cfg.model_cfg
    ckpt = Path(args.checkpoint)
    assert ckpt.is_file(), ckpt
    model_cfg.ckpt_3 = str(ckpt)
    print()
    print('======== LOAD FROZEN AFFECTGPT ========')
    model_cls = registry.get_model_class(model_cfg.arch)
    model = model_cls.from_config(model_cfg)
    validate_model(model)
    for p in model.parameters():
        p.requires_grad = False

    def report_cuda_memory(tag):
        free, total_mem = torch.cuda.mem_get_info(0)
        allocated = torch.cuda.memory_allocated(0) / 1024 ** 3
        reserved = torch.cuda.memory_reserved(0) / 1024 ** 3
        print(f'{tag}: freeGB={free / 1024 ** 3:.2f} totalGB={total_mem / 1024 ** 3:.2f} allocatedGB={allocated:.2f} reservedGB={reserved:.2f}')
    print()
    print('======== CACHED-AV MEMORY PRUNE ========')
    report_cuda_memory('before_prune')
    retained_llama_model = model.llama_model
    removed_child_modules = []
    for child_name, child_module in list(model.named_children()):
        if child_name == 'llama_model':
            continue
        removed_child_modules.append(child_name)
        setattr(model, child_name, None)
    del child_module
    model.llama_model = retained_llama_model
    del retained_llama_model
    gc.collect()
    torch.cuda.empty_cache()
    print('removed unused child modules:', removed_child_modules)
    report_cuda_memory('after_prune')
    llama_dtype_before = next(model.llama_model.parameters()).dtype
    print('llama dtype before move:', llama_dtype_before)
    if llama_dtype_before == torch.float32:
        print('converting frozen LLM from float32 to float16')
        model.llama_model = model.llama_model.half()
        gc.collect()
        torch.cuda.empty_cache()
    llama_dtype_after = next(model.llama_model.parameters()).dtype
    assert llama_dtype_after in {torch.float16, torch.bfloat16}, llama_dtype_after
    print('llama dtype used:', llama_dtype_after)
    model.llama_model = model.llama_model.to(device).eval()
    gc.collect()
    torch.cuda.empty_cache()
    report_cuda_memory('after_llm_move')
    print('CACHED_AV_PRUNE: PASS')
    model.llama_model.config.use_cache = False
    hidden = 3584
    task_token = nn.Parameter(torch.zeros(1, 1, hidden, dtype=torch.float32, device=device))
    torch.nn.init.normal_(task_token, mean=0.0, std=0.02)
    print()
    print('======== TRAINABLE ========')
    print('task_token', tuple(task_token.shape), task_token.numel())
    assert task_token.numel() == 3584
    assert not any((p.requires_grad for p in model.parameters()))
    print('AFFECTGPT TRAINABLE PARAMS: 0')
    print('GENERIC TASK TOKEN PARAMS: 3584')
    print('TASK-ONLY TRAINABLE: PASS')
    builder = TaskPromptBuilder(model, device, task_token)
    optimizer = torch.optim.AdamW([task_token], lr=args.lr, weight_decay=0.0)
    out_dir = WORKDIR / 'checkpoints' / 'gate4' / fold_text('gate4_affectgpt_task_adapter_grouped_outer01_label_fix_v1') / f'seed{args.seed:03d}'
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_metadata = {'protocol': fold_text('gate4_affectgpt_task_adapter_grouped_outer01_label_fix_v1'), 'source_script': Path(__file__).name, 'train_subjects': list(train_subjects_v4), 'validation_subject': development_subject_v4, 'outer_test_subjects': list(outer_test_subjects), 'permanently_excluded_subjects': list(permanently_excluded_subjects), 'train_trials': len(train_rows), 'val_trials': len(val_rows), 'outer_test_trials_used_for_training': 0, 'outer_test_trials_used_for_selection': 0, 'emotion_names': ['neutral', 'anger', 'happiness', 'sadness', 'calmness'], 'label_semantics_fix': 'cache numeric IDs preserved; target words corrected', 'original_source_sha256': 'f6e8068cbe3793eea9456528df8d1040e4342856ac786faa4e4b51958cecf3a6'}
    print()
    print('======== INITIAL VAL ========')
    initial_val = evaluate(model, builder, val_loader)
    print(f'val_loss={initial_val:.6f}')
    history = []
    best_val = float('inf')
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, args.epochs + 1):
        print()
        print('=' * 80)
        print(f'EPOCH {epoch}/{args.epochs}')
        print('=' * 80)
        model.llama_model.eval()
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        seen = 0
        n_batches = len(train_loader)
        for step, batch in enumerate(train_loader, start=1):
            loss = forward_loss(model, builder, batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f'non-finite loss: epoch={epoch}, step={step}')
            raw = float(loss.detach().cpu())
            b = int(batch['label'].shape[0])
            total += raw * b
            seen += b
            (loss / args.grad_accum).backward()
            bad = [name for name, p in model.named_parameters() if p.grad is not None]
            assert not bad, f'Frozen backbone received gradients: {bad[:10]}'
            do_step = step % args.grad_accum == 0 or step == n_batches
            if do_step:
                assert task_token.grad is not None
                torch.nn.utils.clip_grad_norm_([task_token], 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if step == 1 or step % 100 == 0 or step == n_batches:
                free, total_mem = torch.cuda.mem_get_info(0)
                print(f'epoch={epoch} step={step:04d}/{n_batches:04d} loss={raw:.6f} avg={total / seen:.6f} freeGB={free / 1024 ** 3:.2f}')
        train_loss = total / seen
        val_loss = evaluate(model, builder, val_loader)
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        print()
        print(f'EPOCH {epoch} RESULT:')
        print(f'train_loss={train_loss:.6f}')
        print(f'val_loss={val_loss:.6f}')
        print(f'peak_allocated_GB={peak:.2f}')
        row = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'peak_allocated_gb': peak}
        history.append(row)
        torch.save({'task_token': task_token.detach().cpu().float(), 'epoch': epoch, 'val_loss': val_loss, 'seed': args.seed, 'physiology_used': False, 'va_used': False, **checkpoint_metadata}, out_dir / f'epoch{epoch:02d}.pt')
        if val_loss < best_val:
            best_val = val_loss
            torch.save({'task_token': task_token.detach().cpu().float(), 'epoch': epoch, 'val_loss': val_loss, 'seed': args.seed, 'physiology_used': False, 'va_used': False, **checkpoint_metadata}, out_dir / 'best.pt')
            print('BEST TASK TOKEN UPDATED')
        metrics = {'stage': fold_text('stage_a_generic_task_adapter_grouped_outer01'), 'protocol': fold_text('gate4_affectgpt_task_adapter_grouped_outer01_label_fix_v1'), 'seed': args.seed, 'train_subjects': list(train_subjects_v4), 'val_subject': development_subject_v4, 'validation_subject': development_subject_v4, 'outer_test_subjects': list(outer_test_subjects), 'permanently_excluded_subjects': list(permanently_excluded_subjects), 'train_trials': len(train_rows), 'val_trials': len(val_rows), 'outer_test_trials_used_for_training': 0, 'outer_test_trials_used_for_selection': 0, 'physiology_used': False, 'va_used': False, 'trainable': 'single_generic_task_token', 'trainable_params': 3584, 'lr': args.lr, 'batch_size': args.batch_size, 'grad_accum': args.grad_accum, 'epochs': args.epochs, 'initial_val_loss': initial_val, 'history': history, 'emotion_names': ['neutral', 'anger', 'happiness', 'sadness', 'calmness'], 'label_semantics_fix': 'cache numeric IDs preserved; target words corrected', 'original_source_sha256': 'f6e8068cbe3793eea9456528df8d1040e4342856ac786faa4e4b51958cecf3a6'}
        (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print()
    print('=' * 80)
    print(fold_text('GATE4 GROUPED OUTER01 STAGE-A TASK ADAPTER: PASS'))
    print('=' * 80)
    print('initial val loss:', initial_val)
    print('best val loss:', best_val)
    print('checkpoint:', out_dir / 'best.pt')
if __name__ == '__main__':
    main()
