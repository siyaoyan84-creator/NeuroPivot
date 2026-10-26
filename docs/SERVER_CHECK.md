> Current status (2026-10-24): server minimal real initialization, cached-AV scoring and training-side residual backward passed; task-token gradient was disabled. Earlier unexecuted statements below describe the pre-handoff state. See VALIDATION.md for the current verified scope; six historical provenance gaps remain.

# 服务器最小真实检查：一次初始化，不训练

本轮到此结束代码整理。此包用于用户自行上传已有服务器，既不连接服务器，也不安装包、下载模型、覆盖原项目、更新权重或重跑完整实验。

## 包内容与用途

本地交付：`server_delivery/NeuroPivot-server-check-20261023.tar.gz` 及同名 `.sha256`。包内根目录为 NeuroPivot-public，包含当前源码、配置、四文件 AffectGPT patch、许可、必要脚本和说明；不含虚拟环境、外部源码树、数据、权重、缓存、日志、release_checks或私有证据。无需覆盖服务器原 NeuroPivot/AffectGPT 项目。

- `scripts/check_affectgpt_real.py --mode preflight`：只读文件/配置/安全checkpoint元数据和指定trial；不构造任何模型。
- `--mode frozen`：1–2个trial的冻结五类候选NLL，可用outer-test；全程inference_mode，不拟合统计量、不选择checkpoint、不求梯度。
- `--mode gradient`：一个训练侧trial的cross-fitted physiology；临时模型/适配器状态下backward，无optimizer或step。复用已训练权重只验证梯度链路，不复现训练过程。
- `--mode both`：只初始化一次真实模型，分别执行上述冻结评分与训练侧梯度；`--check-task-gradient`才额外检查同一训练侧trial的task-token梯度。
- `scripts/inspect_release_provenance.py`：独立只读来源检查；不从目录名、硬编码metadata或bridge summary推断论文最终链路。

检查器对主模型的LoRA/投影/融合缺项和多余键失败关闭。基础权重的真实from_pretrained加载诊断逐项记录；除可重建position_ids外的缺失/多余/形状问题停止检查。AffectGPT部分checkpoint中缺失、但已由显式基础模型目录加载的冻结主干，与允许缺失且最终路线未使用的新增2D physio_projector分别报告，不混为“全部加载成功”。task/residual关键字段、类别、权重形状和分组必须存在；旧资产未记录的来源列为未知，不伪造来源。

## 1. 上传后在Home独立解压并激活已有环境

将tar.gz和sha256文件上传到Home。以下命令可直接复制；不安装或升级包：

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

如果conda不在PATH，使用实际安装的conda base目录来source其etc/profile.d/conda.sh；不要猜测服务器安装路径。不要运行pip install。脚本的环境检查会报告版本、缺失依赖与实际可见GPU，缺依赖即停，不擅自修复环境。没有默认物理GPU编号：沿用调度器的CUDA_VISIBLE_DEVICES；必要时用户自行指定DEVICE，例如由调度器分配的CUDA设备标识。不要照抄他人的物理卡号。

## 2. 用户只需填写7个已有资产路径

所有路径填写真实绝对路径；下列是参数模板，不声称这些路径已存在。MODEL_ROOT是三个基础模型子目录的父目录；AFFECTGPT_ROOT直接包含my_affectgpt。

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

原始REFED/EAV路径不需要再填写：本次是已缓存真实trial的检查，不执行原始媒体编码或EEG预处理。默认选择由实际分组代码派生：fold1冻结subject14、trial0和1；梯度subject03、trial0，其teacher必须为inner4/job14，且nested_train/val不得包含03。其他fold同样从配置推导，不按被试编号猜身份。REFED anchor编号和EAV编号属于不同命名空间。

不确定路径时，先填写实际项目/运行根目录，再限定范围查找；不要自动取第一条结果：

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

这些find只定位候选文件。由用户确认实际用于该折的主方法文件；检查器依据保存字段/分组校验，不能仅凭目录名确认。若运行根使用其他布局，相应更改find的起始目录；不需要下载runtime全树。

## 3. 先执行只读预检查，再执行最小真实检查

```bash
bash scripts/server_release_check.sh preflight
# Read the printed report.json, environment.txt and provenance_unknown fields first.
bash scripts/server_release_check.sh real
```

每次在解压目录的release_checks中新建唯一子目录，不覆盖旧报告。real模式只有一次真实模型初始化、两个冻结trial五类评分、一个训练侧trial的residual backward；默认不做task-token backward。DEVICE默认cuda表示用户当前设备，不是硬编码物理编号。可显式export DEVICE为已分配设备。

直接CLI的用途是分开使用frozen/gradient或改trial；所有参数见：

```bash
python scripts/check_affectgpt_real.py --help
```

结果包括实际路径、有效配置、依赖/GPU、基础和部分checkpoint加载缺项/多余键、特殊token、形状/有限值、candidate NLL、可训练参数与冻结梯度、参数值版本及已有文件size/mtime/inode前后对比。读取会改变操作系统访问时间，因此不比较atime；这不是所有资产的密码学证明。HF/Torch/CUDA临时缓存也定向到新release_checks目录。任何错误在报告中保留并非零退出；没有生产替身或不安全pickle fallback。

报告包含服务器私有绝对路径，留在服务器检查目录；不要直接发布这些输出或把它们打进源码包。

## 4. 只读核对论文最终链路

桥接证据仅确认：fold1–4的24个job为REFED anchor02，fold5的6个job为anchor01。它不证明最终pivot/task/residual/evaluation使用了这些产物。旧fold5缓存曾硬编码anchor02，必须与真实权重/生成记录比对，不能据此覆盖或修改已有缓存。

先定位确实用于论文的最终JSON/NPZ，不自动选文件。可用下列限定搜索：

```bash
find "$RUN_ROOT/predictions" "$RUN_ROOT/audits/gate4" -maxdepth 3 -type f \( -name '*.json' -o -name '*.npz' \) -print
rg -l 'teacher_checkpoint|physiology_cache|task_checkpoint|base_checkpoint|provenance' "$RUN_ROOT/audits/gate4" "$RUN_ROOT/checkpoints/gate4" -g '*.json'
```

不存在的起始目录请删除或换成实际目录，不创建假路径。填写真实评估文件后运行：

```bash
export FINAL_EVAL="/actual/final-evaluation.json-or-npz"
python scripts/inspect_release_provenance.py --fold 1 --anchor 2 --pivot "$PIVOT_ROOT/subject03.npz" "$PIVOT_ROOT/subject14.npz" --checkpoint "$TASK_CKPT" "$RESIDUAL_CKPT" --evaluation "$FINAL_EVAL" --trusted-checkpoints --output-dir "$PWD/release_checks/provenance-$(date -u +%Y%m%dT%H%M%SZ)"
```

需要查看相邻的真实训练metrics/启动JSON时，用--record显式补充小文件。工具只读NPZ头/小型元数据、JSON来源字段和weights_only+meta/mmap checkpoint；不打印完整数组，不关闭allow_pickle/weights_only。旧checkpoint无法安全读取时，由所有者从可信来源导出小型来源JSON供--record检查，不在工具中放开pickle。

若字段未保存，结论就是“无法从现有记录确定”。最少人工确认：①论文各折/表格确切采用哪个评估文件；②其base/task/residual与AV/pivot输入的实际路径和hash或不可变run ID；③缺失时确认task/residual拟合输入、pivot实际teacher/anchor及生成记录。不能拿目录名或当前配置补写历史证据。

## 检查完成后的停止点

本地已经执行一次真实现有资产的文件/配置预检查，模型初始化次数为0，文件stat未变；原有核心Torch验证不重复。服务器真实forward/backward仍由用户运行。源码可审查、完整新用户运行、论文结果复现是不同状态。即使最小真实检查通过，项目自有部分现适用根目录LICENSE，仍需、确认适用第三方NOTICE/限制并核对论文最终来源；不能宣称完整实验已复现。本轮不push、不公开发布，完成交付后停止重复重构/测试。

subject03属于outer01训练侧，使用inner4/job14的cross-fitted physiology；job14是角色任务ID，不是EAV被试14。REFED与EAV编号属于不同命名空间。
