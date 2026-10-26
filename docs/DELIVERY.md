> Current status (2026-10-24): server minimal real initialization, cached-AV scoring and training-side residual backward passed; task-token gradient was disabled. Earlier unexecuted statements below describe the pre-handoff state. See VALIDATION.md for the current verified scope; six historical provenance gaps remain.

# 服务器最小检查交付（代码整理阶段结束）

本轮交付包：`server_delivery/NeuroPivot-server-check-20261023.tar.gz`，另有SHA256校验文件。准确Linux命令集中在 [SERVER_CHECK.md](SERVER_CHECK.md)，README已收紧为安装、固定源码/patch、数据、主方法、接口与验证范围。没有连接服务器、安装服务器包、加载本地7B、重跑完整实验或push。

## 本轮实际完成

1. 冻结评分与训练侧梯度用途分离；outer01冻结subject14的trial0/1，梯度subject03的trial0，后者必须来自inner4/job14。实际分组推导，不猜测身份。
2. 服务器入口接受显式源码/模型/三个checkpoint/两类缓存/fold/anchor/subject/trial/输出路径。preflight不构造模型；真实模式只初始化一次，最多两条trial候选评分和一条训练侧梯度，task梯度可选。没有optimizer.step或checkpoint保存，复用已训练权重仅诊断梯度链路。
3. 实际加载路径、配置、state_dict缺失/多余键、允许新增physio与外部冻结主干、特殊token、形状/有限值与梯度写入独立release_checks。主模型关键缺项失败关闭；安全读取失败不放开pickle。
4. 只读溯源工具列出JSON/NPZ/安全checkpoint真实保存的来源字段，不扫描打印数组，不把bridge结论扩大为论文最终结果。真实本地预检查通过、初始化次数0、资产stat未变；五份真实来源记录读取无错，但最终论文链仍未确定。
5. 包只含公开源码、配置、patch和必要文件，不含环境、模型、缓存、日志、原始项目、第三方完整树、release_checks输出或私有证据。Home新目录解压，不覆盖服务器项目。服务器现有neuro-affect-g4-affectgpt环境只激活、不安装升级。

## 用户最少填写

AFFECTGPT_ROOT、MODEL_ROOT、BASE_CKPT、TASK_CKPT、RESIDUAL_CKPT、AV_ROOT、PIVOT_ROOT共7个现有路径；示例FOLD=1/ANCHOR=2。GPU沿用调度器分配，无固定物理编号。FINAL_EVAL只在核对论文最终评估来源时额外填写。

## 已确认与未确认

已确认30个bridge job：fold1–4为anchor02，fold5为anchor01，分组/seed/mode一致。REFED/EAV编号命名空间独立。尚未确认的是论文最终实际使用的pivot/task/residual/evaluation来源链接；旧缓存硬编码metadata不能代替实际生成记录。现有来源字段缺失时结论保留“无法从现有记录确定”。

## 真实检查通过后的发布事项

由所有者确认论文最终资产身份/来源、决定NeuroPivot许可证并保留第三方适用署名/NOTICE/非商业等限制；对README中的完整安装和全链运行资格作诚实说明。最小前向/梯度检查通过仍不是完整训练复现，更不是论文指标复现。完整实验不在本轮执行。

代码整理至此收尾；不继续重复重构或重复核心测试。原先核心CPU真实检查与11/11测试结果保留在VALIDATION.md，服务器GPU结果仍待用户运行。

## Final pending release (2026-10-24)

Use the final NeuroPivot-release source tree. The ignored server_delivery/ archive dated20261023 is the preserved pre-server handoff, not the current publication snapshot. Exclude server_delivery/, release_checks/, bytecode and all ignored assets when publishing. Server feedback remains outside this directory. JSON regression passed; see VALIDATION.md. Unresolved: excluded contributor/patch rights, component NOTICE scope and final paper artifact lineage. No push/publication performed.
