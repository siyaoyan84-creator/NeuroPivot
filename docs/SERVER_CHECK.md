> Current status (2026-10-24): server minimal real initialization, cached-AV scoring and training-side residual backward passed; task-token gradient was disabled. Earlier unexecuted statements below describe the pre-handoff state. See VALIDATION.md for the current verified scope; six historical provenance gaps remain.

# Minimal real server check: one initialization, no training

This handoff procedure uses existing server assets. It does not install packages, download models, overwrite the original project, update weights or run full experiments.

## Package and modes

The historical handoff archive contains source, configuration, the four-file AffectGPT patch, licenses and scripts. It excludes environments, external source trees, data, weights, caches, logs, release_checks and private evidence. The historical archive is not included in this current source release.

- scripts/check_affectgpt_real.py --mode preflight reads files, configuration, safe checkpoint metadata and selected trials; it constructs no model.
- --mode frozen scores 1-2 trials over five candidates under inference_mode. Outer-test subjects are allowed; no statistics fitting, checkpoint selection or gradients.
- --mode gradient uses one training-side trial with cross-fitted physiology. Backward uses temporary model/adapter state, without an optimizer or step. Existing trained weights validate the gradient chain, not training reproduction.
- --mode both initializes one real model and runs frozen scoring and training-side residual gradients. --check-task-gradient optionally checks the same training trial task-token gradient.
- scripts/inspect_release_provenance.py reads recorded sources without inferring final paper lineage from directory names, hard-coded metadata or bridge summaries.

Critical LoRA/projection/fusion missing keys and unexpected keys fail the check. Actual foundation from_pretrained diagnostics are recorded; only regenerable position_ids may be missing. Externally loaded frozen foundation keys and allowed unused new physio_projector keys are classified separately. Task/residual key fields, labels, shapes and grouping are required. Historical sources not recorded in old assets remain unknown.

## 1. Extract under Home and activate the existing environment

Upload the historical handoff archive and its checksum to Home before these commands. Do not install or upgrade packages.

```bash
ARCHIVE="$HOME/NeuroPivot-server-check-20261023.tar.gz"
(cd "$HOME" && sha256sum -c NeuroPivot-server-check-20261023.tar.gz.sha256)
CHECK_HOME="$(mktemp -d "$HOME/NeuroPivot-check.XXXXXX")"
tar -xzf "$ARCHIVE" -C "$CHECK_HOME"
cd "$CHECK_HOME/NeuroPivot-public"
bash -n scripts/server_release_check.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate neuro-affect-g4-affectgpt
export PYTHONPATH="$PWD/src"
export PYTHONDONTWRITEBYTECODE=1
```

If conda is not in PATH, source conda.sh from the actual installation. Do not run pip install. The launcher reports versions, missing dependencies and visible GPUs, and stops if dependencies are missing. It inherits scheduler CUDA_VISIBLE_DEVICES without selecting a physical GPU. Set DEVICE only to an allocated logical device.

## 2. Set seven existing asset paths

Use actual absolute paths. These are templates, not claimed existing files. MODEL_ROOT contains the three foundation model directories; AFFECTGPT_ROOT directly contains my_affectgpt.

```bash
export AFFECTGPT_ROOT="/actual/modified/AffectGPT"
export MODEL_ROOT="/actual/existing/models"
export BASE_CKPT="/actual/existing/AffectGPT-base-checkpoint.pth"
export TASK_CKPT="/actual/fold1/task/best.pt"
export RESIDUAL_CKPT="/actual/fold1/va_pivot_real_seed042/final.pt"
export AV_ROOT="/actual/existing/AV-cache-directory"
export PIVOT_ROOT="/actual/fold1/cross-fitted-pivot-directory"
export FOLD=1
export ANCHOR=2
export TRUSTED_CHECKPOINTS=1
# Optional: use only if a separate task-token gradient diagnostic is needed.
export CHECK_TASK_GRADIENT=0
```

REFED/EAV source paths are not required for this cached-trial check; raw media encoding and EEG preprocessing are not run. Subjects derive from actual folds: fold1 frozen subject14 trials0/1 and gradient subject03 trial0. Its teacher is inner4/job14; subject03 must not occur in nested_train/val. Other folds derive subjects from the same protocol. REFED and EAV IDs are separate namespaces.

To locate unknown paths, set the actual project/run root and perform only scoped searches. Do not automatically choose the first result.

```bash
export PROJECT_ROOT="/actual/existing/neuro-affect-pivot"
export RUN_ROOT="/actual/existing/run-or-output-root"
find "$PROJECT_ROOT/third_party/AffectGPT" -maxdepth 4 -type d -name my_affectgpt -print
find "$AFFECTGPT_ROOT/models" -maxdepth 2 -type f -name config.json -print
find "$AFFECTGPT_ROOT/output" -maxdepth 4 -type f -name 'checkpoint_000030_loss_0.751.pth' -print
find "$RUN_ROOT/checkpoints/gate4" -maxdepth 5 -type f -name best.pt -path '*task_adapter_grouped_outer01_label_fix_v1*' -print
find "$RUN_ROOT/checkpoints/gate4" -maxdepth 5 -type f -name final.pt -path '*/va_pivot_real_seed042/*' -print
find "$RUN_ROOT/features/gate4" -maxdepth 2 -type f \( -name subject03.npz -o -name subject14.npz \) -print
```

These searches locate candidates only. Confirm the actual asset for the selected fold/subject and check saved fields and alignment. Adapt search roots to your directory layout; do not transfer the entire runtime tree.

## 3. Run read-only preflight before the minimal real check

```bash
bash scripts/server_release_check.sh preflight
# Read the printed report.json, environment.txt and provenance_unknown fields first.
bash scripts/server_release_check.sh real
```

Each run creates a unique release_checks subdirectory. Real mode performs one initialization, two frozen five-candidate trials and one training-side residual backward. Task-token backward is disabled by default. DEVICE defaults to the current CUDA device without hard-coding a physical GPU.

Use the direct CLI to separate frozen/gradient modes or select different trials:

```bash
python scripts/check_affectgpt_real.py --help
```

Reports include loaded paths, configuration, dependencies/GPU, missing/unexpected keys, special tokens, shapes, finite values, candidate NLL, trainable/frozen gradients, parameter versions and asset size/mtime/inode comparisons. Access time is excluded. This is not a cryptographic proof of all assets. HF/Torch/CUDA caches are redirected to the new check directory. Errors remain in the report and exit nonzero; no production doubles or unsafe pickle fallback.

Reports contain private server paths. Keep them on the server; do not publish or include them in a source archive.

## 4. Inspect final paper lineage read-only

Bridge summaries confirm anchor02 for fold1-4 (24jobs) and anchor01 for fold5 (6jobs). This does not prove the final pivot/task/residual/evaluation consumed these outputs. Old fold5 cache metadata hard-coded anchor02; compare real weights/generation records without overwriting existing caches.

Identify the actual final evaluation JSON/NPZ manually, using scoped searches:

```bash
find "$RUN_ROOT/predictions" "$RUN_ROOT/audits/gate4" -maxdepth 3 -type f \( -name '*.json' -o -name '*.npz' \) -print
rg -l 'teacher_checkpoint|physiology_cache|task_checkpoint|base_checkpoint|provenance' "$RUN_ROOT/audits/gate4" "$RUN_ROOT/checkpoints/gate4" -g '*.json'
```

Remove nonexistent search roots or replace them with actual paths. Set the real evaluation file before running:

```bash
export FINAL_EVAL="/actual/final-evaluation.json-or-npz"
python scripts/inspect_release_provenance.py --fold 1 --anchor 2 --pivot "$PIVOT_ROOT/subject03.npz" "$PIVOT_ROOT/subject14.npz" --checkpoint "$TASK_CKPT" "$RESIDUAL_CKPT" --evaluation "$FINAL_EVAL" --trusted-checkpoints --output-dir "$PWD/release_checks/provenance-$(date -u +%Y%m%dT%H%M%SZ)"
```

Use --record to add explicit small training metrics/launch JSON files. The tool reads NPZ headers/small metadata, JSON source fields and weights_only+meta/mmap checkpoints; it neither prints full arrays nor disables safety restrictions. If safe reading fails, the owner can export a small provenance JSON from a trusted source; do not enable unsafe pickle in this tool.

If fields were never saved, the conclusion is cannot determine from existing records. Minimum human confirmation: the exact evaluation asset used for each paper fold/table; its actual base/task/residual and AV/pivot paths plus hashes or immutable run IDs; and any missing fitting-input, teacher/anchor or generation records. Do not manufacture history from folder names or current configuration.

## Verification boundary

The supplied server minimal real check has passed; see VALIDATION.md for the measured scope. Source reviewability, a new-user end-to-end run and paper-result reproduction are separate claims. Even a passing minimal forward/backward does not establish complete training or paper metrics. LICENSE and LICENSE_SCOPE.md define the project grant and exclusions. Historical nested third-party notices and final result provenance remain unresolved.

Subject03 is training-side for outer01 and uses inner4/job14 cross-fitted physiology. Job14 is a role ID, not EAV subject14. REFED and EAV subject IDs belong to distinct namespaces.
