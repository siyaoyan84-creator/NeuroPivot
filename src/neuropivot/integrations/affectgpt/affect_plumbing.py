from neuropivot.paths import REPO, WORKDIR, OUTER_FOLD, OUTER_FOLDS, TRAIN_SUBJECTS_PUBLIC, N_TRAIN, N_TRIALS, fold_text
from pathlib import Path
import os

import torch
import numpy as np
from neuropivot.integrations.affectgpt.backend import Config, registry
EMOTIONS = ['neutral', 'anger', 'happiness', 'sadness', 'calmness']

class Builder:

    def __init__(self, model, device, task_token, residual_adapter):
        self.model = model
        self.device = device
        self.task_token = task_token
        self.adapter = residual_adapter
        self.tok = model.llama_tokenizer
        self.prompt = '###Human: <Multi><MultiHere></Multi> <Audio><AudioHere></Audio> <Video><FrameHere></Video> <Task><PhysioHere></Task> Classify the emotion as neutral, sadness, anger, happiness, or calmness. Answer with exactly one emotion word. ###Assistant: '
        self.prompt_ids = self.tok(self.prompt, add_special_tokens=False)['input_ids']
        ids = torch.tensor(self.prompt_ids)
        checks = {'MULTI': model.MULTI_PATCH_TOKEN_ID, 'AUDIO': model.AUDIO_PATCH_TOKEN_ID, 'FRAME': model.FRAME_PATCH_TOKEN_ID, 'TASK_SLOT': model.PHYSIO_PATCH_TOKEN_ID}
        print()
        print('======== PROMPT TOKENS ========')
        for name, token_id in checks.items():
            n = int((ids == token_id).sum().item())
            print(name, token_id, 'count=', n)
            assert n == 1

    def build(self, batch, use_residual=True):
        ys = batch['label'].cpu().numpy().astype(int)
        all_ids = []
        all_labels = []
        for y in ys:
            target_ids = self.tok(EMOTIONS[y], add_special_tokens=False)['input_ids']
            if self.tok.eos_token_id is not None:
                target_ids = target_ids + [self.tok.eos_token_id]
            ids = self.prompt_ids + target_ids
            labs = [-100] * len(self.prompt_ids) + target_ids
            all_ids.append(ids)
            all_labels.append(labs)
        max_len = max((len(x) for x in all_ids))
        pad_id = self.tok.pad_token_id
        if pad_id is None:
            pad_id = self.tok.eos_token_id
        padded_ids = []
        padded_labels = []
        attention = []
        for ids, labs in zip(all_ids, all_labels):
            n_pad = max_len - len(ids)
            padded_ids.append(ids + [pad_id] * n_pad)
            padded_labels.append(labs + [-100] * n_pad)
            attention.append([1] * len(ids) + [0] * n_pad)
        input_ids = torch.tensor(padded_ids, dtype=torch.long, device=self.device)
        labels = torch.tensor(padded_labels, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(attention, dtype=torch.long, device=self.device)
        temp = input_ids.clone()
        for token_id in [self.model.MULTI_PATCH_TOKEN_ID, self.model.AUDIO_PATCH_TOKEN_ID, self.model.FRAME_PATCH_TOKEN_ID, self.model.PHYSIO_PATCH_TOKEN_ID]:
            temp[temp == token_id] = 0
        with torch.no_grad():
            base = self.model.llama_model.model.model.embed_tokens(temp)
        dtype = base.dtype
        audio = batch['audio'].to(self.device, dtype=dtype, non_blocking=True)
        frame = batch['frame'].to(self.device, dtype=dtype, non_blocking=True)
        multi = batch['multi'].to(self.device, dtype=dtype, non_blocking=True)
        B = input_ids.shape[0]
        task = self.task_token.to(self.device, dtype=dtype).expand(B, -1, -1)
        if use_residual:
            z = batch['z'].to(self.device, dtype=torch.float32, non_blocking=True)
            delta = self.adapter(z)
            delta = delta.to(dtype=dtype).unsqueeze(1)
        else:
            delta = torch.zeros(B, 1, multi.shape[-1], dtype=dtype, device=self.device)
        multi_mod = multi + delta
        replacements = [(self.model.MULTI_PATCH_TOKEN_ID, multi_mod), (self.model.AUDIO_PATCH_TOKEN_ID, audio), (self.model.FRAME_PATCH_TOKEN_ID, frame), (self.model.PHYSIO_PATCH_TOKEN_ID, task)]
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
        return (embeds, labels, attention_mask, multi, delta)

def forward_loss(model, builder, batch, use_residual=True):
    embeds, labels, attention, multi, delta = builder.build(batch, use_residual=use_residual)
    out = model.llama_model(inputs_embeds=embeds, attention_mask=attention, labels=labels, return_dict=True, use_cache=False)
    with torch.no_grad():
        multi_norm = multi.float().norm(dim=-1).mean()
        delta_norm = delta.float().norm(dim=-1).mean()
        ratio = delta_norm / (multi_norm + 1e-12)
    return (out.loss, float(ratio.detach().cpu()))

@torch.no_grad()
def evaluate(model, builder, loader, use_residual=True):
    model.llama_model.eval()
    builder.adapter.eval()
    total = 0.0
    ratio_total = 0.0
    n = 0
    for batch in loader:
        loss, ratio = forward_loss(model, builder, batch, use_residual=use_residual)
        b = int(batch['label'].shape[0])
        total += float(loss.detach().cpu()) * b
        ratio_total += ratio * b
        n += b
    return (total / n, ratio_total / n)
