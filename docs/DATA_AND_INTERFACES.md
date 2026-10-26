# Data and interface contracts

## Actual directory patterns

The manifest builders are the executable authority. They expect:

```text
<refed_root>/
  data/
    1/ ... 32/
      EEG_videos.mat        # video_1 ... video_15: [64,T_raw]
      EEG_baselines.mat     # each baseline [64,5000]
      fNIRS_videos.mat      # each video [6,51,T_raw]
      fNIRS_baselines.mat   # each baseline [6,51,238]
  annotations/
    1_label.mat ... 32_label.mat
                             # video_1 ... video_15: [seconds,2]
<eav_root>/
  subject1/ ... subject42/   # subject directories are sorted numerically
    EEG/                     # a uniquely identifiable MAT with [10000,30,200]
    Audio/                   # corresponding speaking-trial audio
    Video/
      001_Trial_01_Listening_Neutral.mp4
      ...                    # filename parser resolves sequence/trial/condition/emotion
```

Do not infer channel names or rescale physical units: neither a named electrode-order list nor authoritative raw signal units were found in the selected implementation. Preserve the native MAT channel indices and values. The source performs no additional physical-unit conversion. User data must reproduce the original dataset conventions.

The manifest is a trial identity map, not a training split. It connects numeric sequence IDs, `trial_group_id`, raw EEG trial index, emotion strings, and media paths. Rebuilding it also rebuilds paths for the user's machine; no historical server paths are distributed. `trial-metadata` extracts exactly the 100 speaking rows per subject, sorted by `eeg_trial_index`, and maps emotion **strings** to the model ID order. A/V caching no longer reads an unused `teacher_va` field from an old teacher cache. It does not create fake VA values.

## EEG preprocessing

| Dataset | Raw tensor / rate | Retained input to encoder | Processing |
|---|---|---|---|
| REFED | [64,T], 1000 Hz | [B,64,200], 200 Hz, 1 second | Original MNE resampling and label-anchored one-second windows |
| EAV | [10000,30,200], 500 Hz | [B,30,500], 100 Hz, 5 seconds | Independent parent-trial fifth-order 3–50 Hz Butterworth filtering/resampling, then four non-overlapping windows per 20-second trial |

The EAV raw storage/emotion presentation order and model label order differ. Keep the manifest's filename/trial mapping. Model class IDs are `0 neutral, 1 anger, 2 happiness, 3 sadness, 4 calmness`. No EAV class is assigned an invented continuous VA label: bridge supervision retains native class CE and the original weak ordinal relations.

The REFED legacy low-level loader also loads preprocessed fNIRS MAT files for shape/sample auditing and discards fNIRS before EEG modeling. This dependency is retained to avoid silently changing loader behavior. fNIRS preprocessing uses its original nominal rate `1000/21`, filtering and 25 Hz resampling; final extra-sample trimming is tied to label length. The required rounding check verifies each trial's discrepancy is zero or one sample.

## Representation API

`EEGEncoder.encode_eeg(eeg)` accepts preprocessed, **not yet channel-standardized** EEG as above. It standardizes with the saved training-channel mean/std and returns:

- `va`: [B,2], dimension order [valence, arousal], sigmoid range [0,1]. REFED targets are `(raw_label - 1) / 254`, preserved exactly.
- `latent`: [B,64], the raw output of the shared affect projector; not class probabilities, not an L2-normalized vector, not a guaranteed disentangled physiological measurement.
- `pivot`: [B,66], `cat([va, latent], dim=1)`.

The dataset-specific EEG encoder feeds a 128D base embedding; the shared 64D projector and VA head are calibrated by REFED and frozen for EAV bridge fitting. Only the EAV encoder and private class head train during bridging. The ordinal loss and prototype coordinate construction are retained from actual code.

For downstream AffectGPT use, average VA and latent independently across each trial's **four** windows, then concatenate. The exact retained `aggregate_trials` rejects missing/duplicate window indices, inconsistent parent IDs/labels, and non-100-trial subjects. Training physiology is exported using its held-out inner model. Outer test physiology uses the outer model. No outer-test subject contributes to bridge fitting or channel-stat fitting.

## Normalization and checkpoint version

| Quantity | Fit scope | Rule | Restore |
|---|---|---|---|
| REFED EEG channel stats | REFED source train subjects | NumPy population mean/std over samples and time; std floor 1e-6 | Public source checkpoint `normalization` |
| EAV EEG channel stats | Exact bridge role's train subjects | Same population channel mean/std; std floor 1e-6 | Public bridge checkpoint `normalization`; cache exporter recomputes from the same train subjects |
| 66D downstream physiology | Aligned cross-fitted downstream training trials only | Per-dimension population mean/std; reject std <= 1e-6; no L2 normalization | Residual checkpoint `feature_mean`, `feature_std` |

No statistics are fitted on validation, outer-test, or incoming inference samples. `Normalization.save/load` uses a version-1 NPZ contract; this utility does not replace or reorder the original training calculations.

New EEG checkpoints retain the original `model_state`, `best_epoch`, and selection fields and add `public_format_version=1`, fixed `model_config` (`embedding_dim=128, shared_dim=64, dropout=0.25`), and train-channel statistics. No optimizer-resume support beyond the original code is claimed. Old checkpoints contain only model/selection state: they require explicit statistics computed from their exact training subjects. The interface raises if these are unavailable; it never fits inference data as a fallback. `EEGEncoder.from_checkpoint` strictly loads the fixed architecture and validates normalization dimensions/dataset.

Task checkpoints retain `task_token [1,1,3584]`, selection epoch/loss, five emotion words, train/validation/outer-test identities and no-physiology metadata. Residual checkpoints retain `adapter`, `feature_mean/std`, `feature_dim=66`, `representation=va_pivot`, `mode=real`, main-condition identity, fixed-epoch fit role, hyperparameters and split metadata. The residual checkpoint's original protocol tag identifies its format; it is not reinterpreted as another architecture.

## AV conditioning API and actual hook

`AVConditioner.condition_av(fused_av, physiology)` accepts `[B,3584]` or `[B,1,3584]` fused AV embeddings and raw `[B,66]` physiology (or the encoding dictionary). It computes the saved feature standardization, passes float32 `z` through the actual bias-free 66→128→3584 GELU adapter, casts delta to AV dtype, and returns `fused_av + delta`.

The original integration changes the `MULTI_PATCH_TOKEN_ID` embedding only. The separate audio and frame embeddings remain available in their original slots. The frozen learned task token fills `PHYSIO_PATCH_TOKEN_ID`. The retained `Builder` performs the same token replacement, casts, `unsqueeze(1)`, and Llama forward route as the source. It accesses `model.llama_model.model.model.embed_tokens`, `llama_tokenizer`, and all four patch token IDs. These are requirements of the **modified** AffectGPT integration, not assumed official APIs.

A standalone conditioned embedding is not itself an emotion prediction. The frozen prompt builder and Llama candidate scoring produce predictions. Five-class evaluation minimizes the mean supervised token NLL for each candidate emotion word, including EOS when supplied by the tokenizer. Accuracy and Macro-F1 use corrected numeric IDs. The exact English prompt is preserved even though its displayed list order differs from numeric-ID order.


## Incremental integration qualification

The modified source is now locally available; see AFFECTGPT_SOURCE_AUDIT.json and THIRD_PARTY_DELIVERY.md. Runtime model construction validates four matching/distinct slot IDs and the real embedding width 3584. These checks do not synthesize tokenizer IDs or require an assumed official model equivalence.

Source checkpoints now record refed_test_subject. Anchor NPZs and bridge checkpoints record the explicit chosen anchor and seed. Public pivot exports use that choice instead of hard-coded 2; physiology loaders and residual checkpoint inference reject mismatch with the workspace contract. Legacy direct EEGEncoder/AVConditioner interfaces remain independent of AffectGPT; callers must select mutually compatible EEG/residual weights, and full pipeline artifact provenance is more strict than a bare tensor shape check.

The internal AffectGPT physio_projector receives 2D VA in a different route and is not used by the final 66D residual builder. Actual five-class task fitting allocates a zero task tensor then initializes it with normal_(mean=0,std=0.02) and freezes every AffectGPT parameter. Residual fitting freezes that token as well. The original prompt text and supervised-EOS mean NLL are retained. Tests under Python 3.12 execute actual Builder/fusion source with explicitly test-only NumPy doubles: they validate array/slot/mask dependence, not torch-autograd or pretrained-model performance.


## Release evidence update (2026-10-23)

All30 actual grouped summaries match the frozen EAV subject partition. REFED anchor IDs are separate from EAV subject IDs: folds1–4 use REFED anchor02, fold5 uses REFED anchor01; no cross-dataset identity pairing is implied. Use separate workspaces and saved anchor contracts. The previous unresolved-anchor note is superseded by this per-fold evidence, not by an assumption of one uniform anchor. Actual source/anchor file paths remain absent from the summaries.

Core interfaces have now passed real CPU PyTorch checks in Python3.12.14, including isolated non-editable installation without AffectGPT imports. Their tensors were synthetic inputs to real networks; no dataset/metric reproduction is implied. The complete LLM/tokenizer/processor path remains unrun. AFFECTGPT_ROOT or --affectgpt-root must now be explicit; implicit sibling discovery is disabled. See VALIDATION.md for exact scope and README.md for the real cached-AV check command.
