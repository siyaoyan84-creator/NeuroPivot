# Validation record / 实际验证记录

Updated2026-10-23. This record supersedes the previous missing-torch/skipped-test status. No model/data download, server connection, full training, push or code publication occurred. Only authorized CPU/core software dependencies were installed in a new environment outside the project; original source and private evidence were not edited.

## Compatible real PyTorch environment

Python3.12.14, torch2.5.1+cpu, NumPy1.26.4, SciPy1.14.1, scikit-learn1.5.2, pandas2.2.3 and PyYAML6.0.2, plus installation/build prerequisites. This is a core-check environment, not a certified full data-processing/AffectGPT lockfile. MNE/h5py and the full AffectGPT stack were not qualified here. Old Python3.8 tests are not used to claim this installation passed.

Executed installation arguments, with private environment paths omitted:

```bash
python -m venv <outside-project-env>
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install --only-binary=:all: --index-url https://pypi.org/simple numpy==1.26.4 scipy==1.14.1 scikit-learn==1.5.2 pandas==2.2.3 PyYAML==6.0.2 wheel
```

Initial torch download timed out; it succeeded with an increased network timeout. The initial dependency mirror was too slow and was replaced with official PyPI. No global package environment was modified.

## Real core results: no skips

Executed from the public directory with PYTHONPATH=src and bytecode writing disabled:

```bash
python scripts/check_core.py --original-root .. --output <temporary-result-json>
python -m unittest discover -s tests -v
```

For the source-dependent tests, AFFECTGPT_ROOT was explicitly set to the actual external modified checkout. This enables original-source comparisons; it is not implicit discovery.

| Actual check | Result |
|---|---|
| Real REFED [2,64,200] / EAV [2,30,500] network forward | VA[2,2],latent[2,64],pivot[2,66], finite and correct concatenation |
| Original/public network, identical state/input, both eval | Maximum absolute error0 for every returned tensor on both datasets |
| EEG checkpoint round trip, both dataset routes | Maximum absolute error0 |
| Training-only66D normalization save/load | Maximum absolute error0; existing forbidden-subject-fit checks pass |
| Real nonzero residual adapter versus original class | Maximum absolute error0 |
| condition_av versus explicit AV+adapter(standardized pivot) | Maximum absolute error0; nonzero delta norm6.8464184 |
| Residual checkpoint/statistics round trip | Maximum absolute error0 |
| Real residual backward | fc1 gradient norm0.0140908482; fc2 gradient norm0.1464817226; both finite/nonzero |
| Trainable adapter count | 467200 |
| Frozen real EEG model and standalone AV/task tensors | Zero frozen EEG parameters with gradients; AV/task gradients absent |
| Existing test suite | 11/11pass, zero skips,40.304seconds |

Synthetic inputs are passed through the real EEG/residual networks. The frozen AV/task tensors in the core check are standalone tensors, not a real AffectGPT model. These core gradient results do not establish frozen LLM gradient behavior. Existing lightweight Builder doubles remain clearly scoped and are not presented as complete model validation. A new script's first execution exposed scripts/neuropivot.py shadowing the package; entrypoints now remove that scripts-directory shadow before importing the installed package, and the real checks passed afterward.

## Independent installation actually executed

Copied the public directory to a new temporary directory outside the original tree, excluding bytecode/work/build artifacts. Cleared PYTHONPATH, AFFECTGPT_ROOT, AFFECTGPT_MODEL_ROOT and NeuroPivot work/fold variables. Executed a **non-editable** `pip install --no-deps --no-build-isolation <copied-public-directory>` in the new environment, using the minimal check dependencies already installed. Built/installed neuropivot0.1.0 successfully.

Ran `python -I` imports and `python -I scripts/check_core.py` from the copied tree. Original project paths were absent from sys.path; core interfaces imported and real forward/checkpoint/backward checks passed. No my_affectgpt module was imported. Calling resolve_checkout without an explicit source failed as intended. This qualifies isolated core APIs, not all dataset commands or the complete requirements/AffectGPT installation. See FINAL_CHECKS.json for the separate independent results.

## Historical jobs and source patch

Read all30 private grouped summaries in place. Folds1–4 use REFED anchor02; fold5 uses anchor01. Every job records seed42/ordinal_correct/frozen anchor/no cross-dataset pairing; role/job ID and subject partitions match public code. No actual source checkpoint or anchor path is serialized; code-derived expectations are labelled as such. SOURCE_MAP.md provides the per-job configuration table without copying private metrics or raw records.

Server root/commit/status/diff commands all succeeded. Official AffectGPT commit API identified fffca794c6792023234565370e3f69f0488aede9 and tree0f4ef8e9ce8c1374b915bdedc8cc1b17231d8291. Local commit/tree/object identity was checked. Exported61 tracked source files into a clean temporary baseline, ran git apply --check and actual application, then compared all61 required files with the modified checkout. Pass; only CRLF/LF normalization used. Patch contains four source files, no bytecode or assets. No necessary untracked imported source was found. Official Git network fetching failed on this host; the documented acquisition commands are not represented as an executed successful clone.

Official LAVIS and ImageBind license texts were fetched with the system HTTPS client after Python HTTPS errors; exact URLs/digests are in the existing source audit. Historical nested source revisions/extra NOTICE remain a bounded provenance limitation; see the single consolidated THIRD_PARTY_DELIVERY/NEED_FROM_SERVER records.

## Actual local weight checks and full AffectGPT boundary

Read the real files, not just directory names. All four Qwen safetensors shards are non-LFS payloads; all339 index tensor names map to the correct shard headers and their offsets fit the file bounds. CLIP safetensors header offsets are valid. HuBERT/CLIP bin files are present with non-LFS headers. The664806630-byte AffectGPT checkpoint is a valid ZIP, all1646 ZIP entries pass CRC, and a model key is present in pickle opcodes inspected without execution. No pickle was executed for this inspection. These checks establish file/container structure, **not all tensor values, architecture compatibility or model correctness**.

This host has no usable CUDA; approximately13.95GiB RAM was free at inspection, below the practical needs of this complete unquantized7B integration with its encoders. No full AffectGPT package/environment initialization, actual tokenizer initialization, AV encoder/LLM forward, candidate NLL or LLM autograd was executed. No repeated double-based result is substituted for these checks.

The real check entry is now split into preflight/frozen/gradient/both. Frozen outer-test scoring uses inference_mode; training-side residual/task gradient diagnostics use a separate correctly cross-fitted trial and cloned existing weights, without optimizer steps or checkpoint writes. See SERVER_CHECK.md for the current flags and exact existing-environment commands. Actual GPU execution remains pending; the previous one-trial combined-purpose example is superseded.


## Final source/CLI sweep

Final syntax/help, README CLI parsing, original fingerprints and privacy/artifact sweep results are recorded in FINAL_CHECKS.json after the final sweep. No full training, real-data pipeline execution or paper result reproduction is claimed. Private environment paths and raw server evidence are intentionally not embedded in these public results.

Final sweep completed:59 Python sources passed syntax/3.10 grammar,15 subcommand help routes and both new-check help routes passed,19 non-help README CLI commands parsed. Evidence-matched one/five-fold plans passed; all30 complete eav_fold objects and recorded ordinal/retrieval parameters match. All30 original-source and61 external-source fingerprints remain unchanged. Generated public bytecode from a read-only inspection was removed before the clean artifact/private-path scan, which returned zero hits.


## Server handoff closure

No core suite or full experiment was rerun this round. A real-existing-asset local preflight passed with zero model initializations, saved train-only statistics and two frozen subject14 trials plus one subject03 cross-fitted trial (outer01 inner4/job14). Input/source/model-file size/mtime/inode checks were unchanged. Six downstream provenance fields were absent and explicitly reported unknown; computation readiness is not historical chain verification.

The read-only provenance tool inspected two real pivot caches, task/residual checkpoints with weights_only+meta/mmap, and one small fit JSON: five records, zero read errors, asset stats unchanged. No final paper evaluation artifact was selected or certified. Reports contain private paths and remain outside the source bundle. No unsafe pickle fallback, complete array printing, model initialization or training occurred.

Only bounded new-entry syntax/help, actual protocol guards and package contents are checked for this handoff. The previous task-token zero-initialization wording was incorrect: production allocates zeros and then normal_(0,0.02). Documentation is corrected; no training initialization code changed. Gradient checks reuse cloned trained weights and do not reproduce that initialization/training process.

## Server handoff packaging (2026-10-23)

Python syntax, CLI help and fold-role guards passed as recorded above. Local Windows has no Bash: run `bash -n scripts/server_release_check.sh` on the server first; shell execution remains unverified locally. No additional core training or GPU checks were run. The archive excludes private evidence, assets, environments and reports. Archive members and extracted CLI help were checked.

## Server minimal real check / 服务器最小真实检查 (2026-10-24)

The complete supplied report was parsed and its internal checks agree with PASS. This is an existing server environment, not a clean installation or training qualification.

- Mode both, fold1 / REFED anchor02; exactly one initialization. Frozen cached-AV scoring: EAV subject14 trial indices0,1 (sequence IDs2,4); five finite candidate NLLs each, range0.1858948320–1.1072897911. Zero trainable parameters and zero parameters with gradients in this phase.
- Residual backward: training-side subject03 trial0/sequence2, inner4/job14 cross-fitted physiology. Loss0.6264830828; delta_ratio0.0154926088; fc1/fc2 gradient norms0.3077369034/0.1470687538, both finite and nonzero. Adapter trainable parameters467200; frozen model parameters with gradients0; task_has_grad=false. Task-token gradient check was disabled, not passed by implication.
- Qwen2ForCausalLM, CLIPModel and HubertModel foundation loads: all missing/unexpected/mismatched/error lists empty. Partial AffectGPT checkpoint:412 keys,1348 missing classified as1344 externally loaded frozen foundation keys plus4 allowed new physio keys; critical missing0, unexpected0. Residual load missing/unexpected0. PHYSIO/MULTI/AUDIO/FRAME IDs151670/151669/151666/151667.
- In-memory parameter values unchanged=true; existing supplied assets size/mtime_ns/inode unchanged=true (access time excluded). Optimizer steps0; no statistics fitted and no checkpoint selection. Saved66D statistics restored.
- Passed environment: Python3.10.21, torch2.4.0+cu121, NVIDIA L20; numpy1.23.5, PyYAML6.0.3, transformers4.49.0, peft0.13.2, decord0.6.0, omegaconf2.3.0, einops0.8.0, timm1.0.15, torchaudio2.4.0+cu121, torchvision0.19.0+cu121.

通过范围仅为真实初始化、cached-AV候选评分与训练侧残差梯度链路；未验证原始媒体编码、完整训练、task-token梯度或论文指标复现。JSON中的旧preflight说明字符串不覆盖本次真实加载诊断；YAML frozen_* 标志也不替代实测梯度状态。

Six historical provenance gaps remain: residual fitted anchor; task checkpoint/checkpoint, base_checkpoint and av_cache; residual physiology_cache and base_checkpoint. Bridge summaries establish folds1–4 anchor02 (24jobs), fold5 anchor01 (6jobs), not the final paper result lineage. No missing historical fields were reconstructed. Original reports, logs and paths remain outside this public tree.

本轮仅合并集合/Path序列化、先序列化后独占创建文件与shell LF；小型回归命令：`python tests/test_report_serialization.py`。没有重新加载7B、重复核心测试或运行训练。

Local merge verification (2026-10-24): `python -B tests/test_report_serialization.py` passed 1 test in0.037s. Checked all scripts/*.sh for absence of CR bytes, .gitattributes LF rule, public source for private path/asset markers, and presence of patch/licenses. No model initialization or previously passed core suite was repeated.
