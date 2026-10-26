# NeuroPivot

REFED continuous valence–arousal supervision → grouped cross-dataset EEG bridge → EAV VA2D + latent64D → physiological pivot66D → five-class AffectGPT task adaptation/freeze → physiological residual → emotion prediction.


## Installation

Use Python >=3.10 and an isolated environment. Core CPU checks passed with Python3.12.14/torch2.5.1+cpu; this is not a certified complete AffectGPT environment. Full integration dependencies are separate in requirements-affectgpt.txt; observed version profiles are in configs/. Choose the appropriate PyTorch build for your platform before resolving the full environment.


```bash
python -m venv .venv
# Activate .venv using your platform's command.
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
neuropivot --help
# Only for a newly prepared AffectGPT environment:
python -m pip install -r requirements-affectgpt.txt
```

For the server check, activate the **existing** neuro-affect-g4-affectgpt environment and install/upgrade nothing. Use [SERVER_CHECK](docs/SERVER_CHECK.md); its launcher uses this unpacked source directly.


## Fixed AffectGPT source + patch

Use the reviewed modified implementation, not an assumed equivalent official release. The following fixed baseline plus four-file patch passed clean-baseline application and all61 required-source comparisons. Network Git acquisition was not successful on this local host. Explicit --affectgpt-root/AFFECTGPT_ROOT is mandatory; implicit sibling discovery is disabled.


```bash
git clone --filter=blob:none --no-checkout https://github.com/zeroQiaoba/AffectGPT.git external/AffectGPT
git -C external/AffectGPT sparse-checkout set --no-cone /AffectGPT/config.py /AffectGPT/my_affectgpt/ /AffectGPT/toolkit/ /AffectGPT/train_configs/ /AffectGPT/LICENSE /AffectGPT/environment.yml
git -C external/AffectGPT -c core.autocrlf=false checkout --detach fffca794c6792023234565370e3f69f0488aede9
git -C external/AffectGPT apply --unidiff-zero --check ../../patches/affectgpt-physio.patch
git -C external/AffectGPT apply --unidiff-zero ../../patches/affectgpt-physio.patch
neuropivot inspect-affectgpt --source external/AffectGPT/AffectGPT
```

AffectGPT Apache text, LAVIS/MiniGPT BSD notices and ImageBind CC-BY-NC-SA restrictions remain separate; see [THIRD_PARTY_DELIVERY](docs/THIRD_PARTY_DELIVERY.md) and licenses/.

**License:** Copyright(c)2026 Siyao Yan. Siyao Yan-authored original content is available for noncommercial research, learning, reproduction, modification and conditional redistribution under [LICENSE](LICENSE). Commercial use requires separate prior written authorization: dangjisheng@lzu.edu.cn. No commercial exemptions apply. This is noncommercial source availability, not OSI open source. Third-party and unclear-authorship portions are excluded; see [LICENSE_SCOPE](docs/LICENSE_SCOPE.md).


## Data and weights

Prepare legally obtained REFED/EAV and valid Qwen2.5-7B-Instruct, CLIP ViT Large, Chinese HuBERT Large and AffectGPT base weights. This source tree distributes no datasets, caches or weights. Train the NeuroPivot source/bridge/task/residual checkpoints before inference. Existing weight filenames or structural checks alone do not establish functional compatibility.


```text
<refed_root>/data/1/ ... 32/   EEG_videos.mat, EEG_baselines.mat,
                             fNIRS_videos.mat, fNIRS_baselines.mat
<refed_root>/annotations/1_label.mat ... 32_label.mat
<eav_root>/subject1/ ... subject42/   EEG/, Audio/, Video/
<model_root>/Qwen2.5-7B-Instruct/
<model_root>/clip-vit-large-patch14/
<model_root>/chinese-hubert-large/
```

Edit configs/paths.example.yaml. All eight output directories must agree with --work-dir. Preserve dataset-native channel order/units; the source does not establish complete electrode names or a new unit conversion. REFED EEG uses [B,64,200] at200Hz/1s; EAV uses [B,30,500] at100Hz/5s after original preprocessing. The REFED loader retains fNIRS alignment checks but feeds only EEG to the pivot network.


## Main-method training and inference

The planner covers manifests/preprocessing, REFED source, anchor, six bridge roles per fold, cross-fitted pivot export, AV export, task fitting, residual fitting and final evaluation. It prints commands by default; --execute explicitly runs them with real prerequisites. Run from the repository root.


```bash
# Set these to your actual prepared assets, not the example names.
export AFFECTGPT_ROOT=/prepared/modified-AffectGPT
export MODEL_ROOT=/prepared/models
export BASE_CKPT=/prepared/affectgpt-base.pth
# Evidence-matched one-fold plan:
python scripts/run_pipeline.py --anchor 2 --folds 1 --paths-config configs/paths.example.yaml --base-checkpoint "$BASE_CKPT" --affectgpt-root "$AFFECTGPT_ROOT" --model-root "$MODEL_ROOT"
# Evidence-matched five folds require two independent workspaces:
python scripts/run_pipeline.py --anchor 2 --folds 1 2 3 4 --paths-config configs/paths.example.yaml --base-checkpoint "$BASE_CKPT" --affectgpt-root "$AFFECTGPT_ROOT" --model-root "$MODEL_ROOT"
python scripts/run_pipeline.py --anchor 1 --folds 5 --work-dir work_fold5 --paths-config configs/paths.fold5.example.yaml --base-checkpoint "$BASE_CKPT" --affectgpt-root "$AFFECTGPT_ROOT" --model-root "$MODEL_ROOT"
```

All30 bridge summaries record anchor02 for folds1–4 and anchor01 for fold5; that does **not** identify the final paper evaluation chain. REFED anchor numbers and EAV participant numbers are separate namespaces. EAV01 is excluded,02 is validation,03 stays training-side,04–42 form outer evaluation. No split or class order was changed.


Individual-stage examples after data/outputs are prepared:


```bash
neuropivot --work-dir work train-refed --paths-config configs/paths.example.yaml --mode separate --refed-test-subject 2 --seed 42 --run-tag pilot
neuropivot --work-dir work export-refed-anchor --paths-config configs/paths.example.yaml --refed-test-subject 2 --seed 42
neuropivot --work-dir work bridge --outer-fold 1 --split-role outer --paths-config configs/paths.example.yaml --mode ordinal_correct --refed-anchor-test-subject 2 --seed 42
# Train outer + inner1 ... inner5 before pivot export:
neuropivot --work-dir work export-pivots --outer-fold 1 --paths-config configs/paths.example.yaml --refed-anchor-test-subject 2
neuropivot --work-dir work --affectgpt-root "$AFFECTGPT_ROOT" --model-root "$MODEL_ROOT" cache-av --subjects 2-42 --cfg-path configs/affectgpt.main.yaml --checkpoint "$BASE_CKPT"
neuropivot --work-dir work --affectgpt-root "$AFFECTGPT_ROOT" --model-root "$MODEL_ROOT" train-task --outer-fold 1 --cfg-path configs/affectgpt.main.yaml --checkpoint "$BASE_CKPT"
neuropivot --work-dir work --affectgpt-root "$AFFECTGPT_ROOT" --model-root "$MODEL_ROOT" train-residual --outer-fold 1 --cfg-path configs/affectgpt.main.yaml --checkpoint "$BASE_CKPT"
neuropivot --work-dir work --affectgpt-root "$AFFECTGPT_ROOT" --model-root "$MODEL_ROOT" predict --outer-fold 1 --cfg-path configs/affectgpt.main.yaml --checkpoint "$BASE_CKPT" --output work/predictions/outer01.npz
```

Task adaptation trains only a [1,1,3584] token in PHYSIO; production initialization is zero allocation followed by normal_(0,0.02). Residual fitting freezes that learned token and AffectGPT, trains only the bias-free66→128→3584 GELU adapter, and adds delta to multi_llms before MULTI replacement. The internal2D physio_projector is not used. Class IDs are neutral,anger,happiness,sadness,calmness. Candidate scoring uses mean supervised-label-token NLL including EOS; outputs include Accuracy,Macro-F1 and label-token NLL.


## Pluggable interfaces

```python
from neuropivot import EEGEncoder, AVConditioner
encoder = EEGEncoder.from_checkpoint("your-trained-bridge-best.pt")
physiology = encoder.encode_eeg(eeg)  # va[B,2], latent[B,64], pivot[B,66]
conditioner = AVConditioner.from_checkpoint("your-trained-residual-final.pt")
conditioned = conditioner.condition_av(fused_av, physiology)  # [B,1,3584]
```

The core API imports no AffectGPT. Inputs are preprocessed EEG before saved channel standardization; pivot concatenates [valence,arousal,latent64]. For downstream trial inference, average the original four windows first and restore saved training-only statistics; never fit inference data. See examples/ and DATA_AND_INTERFACES.md for checkpoint versions and alignment.


## Verified scope

Real CPU core forward/backward, original/public comparisons (maximum absolute error0), checkpoint/statistics round trips and isolated core installation passed; the existing11 tests passed with zero skips. The server minimal real check passed in an existing Python3.10.21 / torch2.4.0+cu121 / NVIDIA L20 environment: one real AffectGPT initialization, two cached-AV trials with finite five-class candidate NLL, and one training-side cross-fitted trial with finite nonzero residual gradients. Frozen model/task parameters had no gradients; no optimizer step, statistics fitting, checkpoint selection or asset stat change occurred. Task-token gradient checking was disabled.


See [VALIDATION](docs/VALIDATION.md) for sanitized measured results and the passed environment, [SERVER_CHECK](docs/SERVER_CHECK.md) for the optional check procedure, and [NEED_FROM_SERVER](docs/NEED_FROM_SERVER.md) for unresolved historical records. The project license covers only Siyao Yan-authored original portions; third-party and unclear-authorship exclusions remain as listed in LICENSE_SCOPE.md.
