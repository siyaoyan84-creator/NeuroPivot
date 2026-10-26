from neuropivot.integrations.affectgpt.backend import validate_model
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import torch
from neuropivot.inference import evaluator as ge
from neuropivot.inference.scoring import metrics, score_candidates

def main():
    p = argparse.ArgumentParser(description='Frozen main-method five-class candidate NLL inference')
    p.add_argument('--cfg-path', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--options', nargs='+', default=['inference.test_epoch=30', 'inference.gpu=0'])
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    affect_checkpoint = Path(args.checkpoint).resolve()
    task_object = ge.load_torch(ge.TASK_CKPT)
    if task_object['train_subjects'] != ge.TRAIN_SUBJECTS or task_object['outer_test_subjects'] != ge.OUTER_TEST_SUBJECTS or task_object['validation_subject'] != 2:
        raise ValueError('Task checkpoint split mismatch')
    _, checkpoint = ge.load_condition('va_pivot_real', 'va_pivot', 'real', 66, 'none')
    rows, _ = ge.prepare_rows('va_pivot', checkpoint, False, 42)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Original frozen inference requires one visible CUDA device')
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = torch.device('cuda:0')
    configuration = ge.stageb.Config(args)
    model_cfg = configuration.model_cfg
    model_cfg.ckpt_3 = str(affect_checkpoint)
    model = ge.stageb.registry.get_model_class(model_cfg.arch).from_config(model_cfg)
    validate_model(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for child_name in list(model._modules):
        if child_name != 'llama_model':
            setattr(model, child_name, None)
    gc.collect()
    torch.cuda.empty_cache()
    if next(model.llama_model.parameters()).dtype == torch.float32:
        model.llama_model = model.llama_model.half()
    model.llama_model = model.llama_model.to(device).eval()
    model.llama_model.config.use_cache = False
    task = task_object['task_token'].float().to(device)
    task.requires_grad_(False)
    if tuple(task.shape) != (1, 1, 3584):
        raise ValueError('Task token shape mismatch')
    adapter = ge.ft.ResidualAdapter(66).to(device).eval()
    adapter.load_state_dict(checkpoint['adapter'], strict=True)
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    builder = ge.stageb.Builder(model, device, task, adapter)
    scores = []
    for row in rows:
        values, counts = score_candidates(ge, model, builder, row, True)
        scores.append(values)
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray([r['label'] for r in rows], dtype=np.int64)
    prediction = scores.argmin(axis=1)
    result = metrics(labels, prediction)
    result['mean_label_token_nll'] = float(scores[np.arange(len(labels)), labels].mean())
    result['nll_definition'] = 'Mean supervised candidate-label tokens including EOS, matching original builder'
    result['outer_test_subjects'] = ge.OUTER_TEST_SUBJECTS
    result['per_subject'] = {str(s): metrics(labels[np.asarray([r['subject'] == s for r in rows])], prediction[np.asarray([r['subject'] == s for r in rows])]) for s in ge.OUTER_TEST_SUBJECTS}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as h:
        np.savez_compressed(h, candidate_nll=scores, label=labels, prediction=prediction, subject_number=np.asarray([r['subject'] for r in rows]), sequence_id=np.asarray([r['sequence_id'] for r in rows]), candidate_token_counts=counts)
    args.output.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
