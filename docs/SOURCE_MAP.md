# Source map and selection evidence

Updated 2026-10-23. Main-method selections follow actual grouped-run calls, cache readers and task/residual checkpoint consumers; version names alone were not used. Only required shared classes from historical multi-condition scripts are retained, with control branches removed. Raw release_evidence summaries, metrics, Git status and export manifests remain private outside this directory.

## Source → public implementation

| Original source | Public implementation |
|---|---|
| `src/models/gate3_eeg_pivot_v2.py` | `src/neuropivot/models/eeg.py` |
| `src/models/gate3_eeg_pivot_v3.py` | `src/neuropivot/models/ordinal.py` |
| `src/data/schema.py` | `src/neuropivot/data/schema.py` |
| `src/data/mat_header.py` | `src/neuropivot/data/mat_header.py` |
| `src/data/eav_manifest.py` | `src/neuropivot/data/eav_manifest.py` |
| `src/data/refed_manifest.py` | `src/neuropivot/data/refed_manifest.py` |
| `src/preprocessing/refed.py` | `src/neuropivot/preprocessing/refed.py` |
| `src/config.py` | `src/neuropivot/config.py` |
| `src/preprocessing/eav.py` | `src/neuropivot/preprocessing/eav.py` |
| `src/samples/refed.py` | `src/neuropivot/samples/refed.py` |
| `scripts/run_refed_light_subject.py` | `src/neuropivot/data/refed_loader.py` |
| `scripts/run_gate3a_refed_target_loso.py` | `src/neuropivot/training/refed_source.py` |
| `scripts/build_gate3a_refed_anchor_v4.py` | `src/neuropivot/training/refed_anchor.py` |
| `scripts/run_gate3a_anchored_pivot_v4_nested_outer03.py` | `src/neuropivot/training/bridge_base.py` |
| `scripts/run_gate3a_anchored_pivot_v4_grouped_outer5fold.py` | `src/neuropivot/training/protocol.py` |
| `scripts/build_gate4_eav_nested_outer03_pivot_cache_v1.py` | `src/neuropivot/inference/pivot_cache_base.py` |
| `scripts/build_gate4_eav_grouped_outer01_pivot_cache_v1.py` | `src/neuropivot/inference/export_pivots.py` |
| `scripts/train_gate4_grouped_outer01_three_conditions_label_fix_v1.py` | `src/neuropivot/models/residual.py` |
| `scripts/train_gate4_affectgpt_physio_residual_v2.py` | `src/neuropivot/integrations/affectgpt/affect_plumbing.py` |
| `scripts/train_gate4_affectgpt_pivot_residual_v4.py` | `src/neuropivot/data/av_physio_data.py` |
| `scripts/train_gate4_grouped_outer01_three_conditions_label_fix_v1.py` | `src/neuropivot/training/residual.py` |
| `scripts/train_gate4_affectgpt_task_adapter_grouped_outer01_label_fix_v1.py` | `src/neuropivot/training/task_adapter.py` |
| `scripts/eval_gate4_grouped_outer01_label_fix_v1.py` | `src/neuropivot/inference/evaluator.py` |
| `scripts/eval_gate4_grouped_fiveclass_label_fix_v1.py` | `src/neuropivot/inference/scoring.py` |
| `scripts/cache_gate4_affectgpt_av_embeddings.py` | `src/neuropivot/integrations/affectgpt/cache_av.py` |
| `scripts/build_manifest.py` | `src/neuropivot/data/build_manifest.py` |
| `scripts/build_refed_preprocessed_cache.py` | `src/neuropivot/data/preprocess_refed.py` |
| `scripts/build_gate3_eav_eeg_cache.py` | `src/neuropivot/data/preprocess_eav.py` |
| `scripts/build_refed_sample_registry.py` | `src/neuropivot/data/refed_samples.py` |
| `scripts/audit_refed_fnirs_resample_rounding.py` | `src/neuropivot/data/refed_rounding.py` |

REFED V2 architecture and V3 ordinal supervision are the actual shared architecture used by the grouped bridge. The retained grouped runner defines fixed subject folds and inner held-subject roles; the cache exporter consumes those six checkpoints. Stage-A label-fix task adaptation and the fixed-three-epoch residual-fit trainer are selected by downstream checkpoint paths and matching five-class label IDs. The residual class imported from three_conditions is necessary shared code; its control conditions are not included.

## Final grouped evidence: all 30 jobs

Every available final grouped summary was read. All recorded seed42, mode ordinal_correct, anchor_frozen=true, cross_dataset_subject_pairing=false, and train/val/test sets identical to the public generator. Actual checkpoint/anchor paths are not recorded in these summaries. The runner constructs them using the explicit REFED anchor argument; those are derived expected paths, not observed saved arguments.

| Fold | Role | Job ID | REFED anchor subject | Seed | Mode | Actual source/anchor path in summary |
|---|---|---:|---:|---:|---|---|
| 1 | outer | 10 | 2 | 42 | ordinal_correct | Not recorded |
| 1 | inner1 | 11 | 2 | 42 | ordinal_correct | Not recorded |
| 1 | inner2 | 12 | 2 | 42 | ordinal_correct | Not recorded |
| 1 | inner3 | 13 | 2 | 42 | ordinal_correct | Not recorded |
| 1 | inner4 | 14 | 2 | 42 | ordinal_correct | Not recorded |
| 1 | inner5 | 15 | 2 | 42 | ordinal_correct | Not recorded |
| 2 | outer | 20 | 2 | 42 | ordinal_correct | Not recorded |
| 2 | inner1 | 21 | 2 | 42 | ordinal_correct | Not recorded |
| 2 | inner2 | 22 | 2 | 42 | ordinal_correct | Not recorded |
| 2 | inner3 | 23 | 2 | 42 | ordinal_correct | Not recorded |
| 2 | inner4 | 24 | 2 | 42 | ordinal_correct | Not recorded |
| 2 | inner5 | 25 | 2 | 42 | ordinal_correct | Not recorded |
| 3 | outer | 30 | 2 | 42 | ordinal_correct | Not recorded |
| 3 | inner1 | 31 | 2 | 42 | ordinal_correct | Not recorded |
| 3 | inner2 | 32 | 2 | 42 | ordinal_correct | Not recorded |
| 3 | inner3 | 33 | 2 | 42 | ordinal_correct | Not recorded |
| 3 | inner4 | 34 | 2 | 42 | ordinal_correct | Not recorded |
| 3 | inner5 | 35 | 2 | 42 | ordinal_correct | Not recorded |
| 4 | outer | 40 | 2 | 42 | ordinal_correct | Not recorded |
| 4 | inner1 | 41 | 2 | 42 | ordinal_correct | Not recorded |
| 4 | inner2 | 42 | 2 | 42 | ordinal_correct | Not recorded |
| 4 | inner3 | 43 | 2 | 42 | ordinal_correct | Not recorded |
| 4 | inner4 | 44 | 2 | 42 | ordinal_correct | Not recorded |
| 4 | inner5 | 45 | 2 | 42 | ordinal_correct | Not recorded |
| 5 | outer | 50 | 1 | 42 | ordinal_correct | Not recorded |
| 5 | inner1 | 51 | 1 | 42 | ordinal_correct | Not recorded |
| 5 | inner2 | 52 | 1 | 42 | ordinal_correct | Not recorded |
| 5 | inner3 | 53 | 1 | 42 | ordinal_correct | Not recorded |
| 5 | inner4 | 54 | 1 | 42 | ordinal_correct | Not recorded |
| 5 | inner5 | 55 | 1 | 42 | ordinal_correct | Not recorded |


Thus folds1–4 are confirmed anchor02 (24 jobs), fold5 anchor01 (6 jobs). The earlier full-LOSO anchor01 spec belongs to another protocol. Grouped cache builders hard-coded anchor02 metadata; this is inconsistent with fold5 execution summaries and is not used to override them. The public exporter writes explicit/checkpoint-verified identity. The planner marks history confirmed only when its chosen anchor agrees with every selected fold; it supports two independent workspaces for the observed mixed-anchor five folds.

REFED and EAV participant numbers are different dataset namespaces; no cross-dataset pairing was performed. EAV01 is excluded,02 is validation,03 remains on the training/cross-fitting side and04–42 form outer evaluation. All30 job splits match this rule. The code expresses the selection rule, not an invented biological justification for it.

## Modified AffectGPT dependency chain

The actual inner source contains my_affectgpt. Config/registry/package registration, YAML arch affectgpt, model.from_config, tokenizer, Chat, MER2025OV inherited data reader and processors are present. Local import closure61files has no unresolved local import. No YAML defaults/include directive is used in the selected user configuration; Config performs its actual OmegaConf merges. The optional missing configs/models/affectgpt.yaml belongs to from_pretrained and is not used by the retained from_config route.

The actual four-file server diff is nonempty and successful. Official baseline commit fffca794c6792023234565370e3f69f0488aede9 was verified, its commit/tree identities checked and clean baseline source files exported. After git apply --check and application of the sanitized text-only patch, all61 required files match the supplied source (CRLF/LF normalization only). All61 are tracked in that baseline; no imported untracked source is needed. Unchanged fusion code is already in the verified baseline, not assumed to be a still-uncommitted modification.

Task token initialized with normal_(mean=0,std=0.02), after initial zero allocation, [1,1,3584] occupies PHYSIO and alone trains StageA. StageB freezes it and AffectGPT; an external bias-free66→128→3584 GELU adapter adds delta to multi_llms before MULTI replacement. The internal two-dimensional physio_projector is not used by this final route. The tokenizer's real special tokens and original prompt/mean supervised-token NLL including EOS are preserved. Actual full-LLM backward is pending; real core torch tests and lightweight Builder checks have separate validation scopes.

## Scope and original-file preservation

No data, pretrained weights, historical results, ablations, search, plotting or significance tests are copied here. Third-party source is obtained via fixed upstream plus verified patch; raw private evidence is not packaged. Original file fingerprints and external dependency fingerprints are retained for verification. See VALIDATION.md and THIRD_PARTY_DELIVERY.md for execution and license boundaries.


Server-check boundary: subject14 is outer01 test and is used only for frozen scoring. Subject03 is in outer01 training-side inner4 held-out physiology/job14; only it is used for gradient diagnostics in the example. This derives from configs/folds.json and the actual grouped generator. The thirty bridge summaries do not establish the final paper's cache/task/residual/evaluation lineage. Read-only inspection of saved fields remains necessary; folder names and hard-coded metadata are not substituted as proof.
