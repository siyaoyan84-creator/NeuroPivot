#!/usr/bin/env bash
# Run from the unpacked NeuroPivot-public directory in the existing environment.
set -euo pipefail
MODE="${1:-preflight}"
case "$MODE" in preflight|real) ;; *) echo 'Usage: bash scripts/server_release_check.sh preflight|real' >&2; exit 2;; esac
for name in AFFECTGPT_ROOT MODEL_ROOT BASE_CKPT TASK_CKPT RESIDUAL_CKPT AV_ROOT PIVOT_ROOT FOLD ANCHOR; do
  if [[ -z "${!name:-}" ]]; then echo "Set explicit $name first" >&2; exit 2; fi
done
if [[ "${TRUSTED_CHECKPOINTS:-}" != 1 ]]; then echo 'Set TRUSTED_CHECKPOINTS=1 only for your own trusted assets; weights_only remains enforced' >&2; exit 2; fi
export PYTHONPATH="$PWD/src"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "${CHECKS_ROOT:-$PWD/release_checks}"
OUT="$(mktemp -d "${CHECKS_ROOT:-$PWD/release_checks}/${MODE}.XXXXXX")"
export FOLD
python - <<'PY' | tee "$OUT/environment.txt"
import importlib.metadata,sys,torch
assert sys.version_info >= (3,10), 'Python >=3.10 required'
print('Python:',sys.version.split()[0], 'torch:',torch.__version__)
print('CUDA:',torch.cuda.is_available(),'visible devices:',torch.cuda.device_count())
for i in range(torch.cuda.device_count()):print('Visible device',i,torch.cuda.get_device_name(i))
missing=[]
for name in ['numpy','PyYAML','transformers','peft','decord','omegaconf','einops','timm','torchaudio','torchvision']:
    try:print(name,importlib.metadata.version(name))
    except importlib.metadata.PackageNotFoundError:missing.append(name)
if missing:raise SystemExit('Missing dependencies (nothing was installed): '+', '.join(missing))
PY
read -r EVAL_SUBJECT GRAD_SUBJECT GRAD_ROLE GRAD_JOB < <(python - <<'PY'
import os
from neuropivot.training.protocol import build_split,job_id_for
fold=int(os.environ['FOLD']);s=build_split(fold,'outer');e=s['test'][0];g=min(s['train'])
role=next(r for r in ['inner1','inner2','inner3','inner4','inner5'] if g in build_split(fold,r)['test'])
print(e,g,role,job_id_for(fold,role))
PY
)
printf -v ES '%02d' "$EVAL_SUBJECT"
printf -v GS '%02d' "$GRAD_SUBJECT"
echo "Frozen subject=$ES; gradient subject=$GS ($GRAD_ROLE/job$GRAD_JOB); fold=$FOLD anchor=$ANCHOR"
COMMON=(--fold "$FOLD" --anchor "$ANCHOR" --affectgpt-root "$AFFECTGPT_ROOT" --model-root "$MODEL_ROOT"
 --cfg-path "$PWD/configs/affectgpt.main.yaml" --checkpoint "$BASE_CKPT" --task-checkpoint "$TASK_CKPT" --residual-checkpoint "$RESIDUAL_CKPT"
 --eval-subject "$EVAL_SUBJECT" --eval-trials 0 1 --eval-av-npz "$AV_ROOT/subject$ES.npz" --eval-physiology-npz "$PIVOT_ROOT/subject$ES.npz"
 --grad-subject "$GRAD_SUBJECT" --grad-trial 0 --grad-av-npz "$AV_ROOT/subject$GS.npz" --grad-physiology-npz "$PIVOT_ROOT/subject$GS.npz"
 --trusted-checkpoints)
if [[ "$MODE" == preflight ]]; then
 python scripts/check_affectgpt_real.py --mode preflight "${COMMON[@]}" --output-dir "$OUT/files"
else
 EXTRA=()
 if [[ "${CHECK_TASK_GRADIENT:-0}" == 1 ]]; then EXTRA+=(--check-task-gradient); fi
 python scripts/check_affectgpt_real.py --mode both "${COMMON[@]}" --device "${DEVICE:-cuda}" "${EXTRA[@]}" --output-dir "$OUT/real"
fi
echo "Private check output: $OUT (never included in source packages)"
