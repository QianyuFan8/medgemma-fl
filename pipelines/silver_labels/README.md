# Ridge teacher：队友一键运行包

此流程用于新的 gold / silver-label 实验。2026-11-17。
真实原始 beta 矩阵仍在队友电脑上；本地完成的是代码、环境、元数据核验和合成数据测试，不是实际患者的模型结果。

## v4：所有训练用途只允许 ERMS / ARMS

**使用本包 v4，不要再使用 v2 的114/97或 v3 的109/81训练名单。**
白名单同时用于 Ridge teacher、MedGemma gold training 和 silver training。其他亚型以及未知/缺失亚型均不参与训练。

| 用途 | 人数 | POS / NEG |
|---|---:|---|
| Ridge teacher training | 108 | 53 / 55 |
| MedGemma gold training | 19 | 10 / 9 |
| Gold validation | 20 | 12 / 8 |
| Gold test | 20 | 12 / 8 |
| Silver training candidates（标签未知） | 79 | 待预测 |

- Ridge training 从 v3 的109人中再移除1位 Infantile；silver训练候选从81人中再移除1位 Sclerosing、1位 Unknown。
- 本轮 gold training 全部为 ERMS/ARMS。Spindle 保留在 validation，RMS-NOS 保留在 test，仅用于评估；v3确认的互换和全部holdout成员不变。
- Ridge人数：**146位ERMS/ARMS gold − 38位ERMS/ARMS holdout = 108位训练患者**。另外2位其他亚型holdout原本就不在146人中；所有40位holdout均不参加任何训练。
- 原97位缺标签paired候选中有18位非ERMS/ARMS（8 Spindle、8 RMS-NOS、1 Sclerosing、1 Unknown），留下79位silver训练候选，之后仍需筛选高置信度预测。
- 另6位非ERMS/ARMS甲基化gold（5 RMS-NOS、1 Infantile）也不训练。以上24位保留在 `inputs/cohort/prediction_only.csv`；本次一键流程不拟合、不伪标为训练数据，也不自动评分他们。
- 所有ERMS/ARMS paired患者共137位：57位同样本gold、79位缺标签候选、1位跨样本标签待审核。这个137人统计不包含上述2位其他亚型holdout。
- `revision_changes.csv`记录v3→v4的3位新增排除，`v3_swap_and_exclusions.csv`记录上次互换及排除；`revision.json`记录新旧manifest哈希。
- 亚型按患者所有样本核查：任何样本出现非ERMS/ARMS，都禁止该患者进入训练。未知、缺失和未来新出现的亚型默认不允许训练。

队友旧队列201条记录、148位患者，经身份审核排除2位后是146位ERMS/ARMS gold。其他8位亚型仍可保留分析，但不进入训练。
旧56人实验结果没有被改写；本包是新的探索性划分。

## 队友只需这样运行

将私有运行包解压到一个普通可写目录，进入 `ridge_teacher_handoff`：

```bash
bash run.sh --data-dir "/path/to/COMET_RMS_Data"
```

**只需替换原始数据文件夹路径。** 无需修改 Python 文件，无需 GPU。
首次运行自动在本文件夹创建独立 Python + R + LIMMA 环境，不改系统环境；需要网络下载依赖。
支持 macOS Intel / Apple Silicon、Linux；Windows 请在 WSL2 Ubuntu 下运行。

原始数据文件夹应包含：

```text
COMET_RMS_Data/
  RMSs_beta.txt                  # 完整原始矩阵，不是旧 849-CpG 导出
  RMSs_meta.txt                  # Sentrix_ID、UID.Subject、Final.QC 等
  COMET_RMS.xlsx                 # Sample ID / Sentrix 配对来源
  COMET_RMS_fusion_status.xlsx   # 原始真实 fusion 标签
```

beta TSV 格式为 `TargetID` + `<Sentrix_ID>.Ave_Beta` 列。
检测 p 值可以是同一文件里的 `<Sentrix_ID>.Detection.Pval` 列，也可以单独提供：

```bash
bash run.sh --data-dir "/path/to/COMET_RMS_Data" \
  --detection-file "/path/to/detection_p.tsv"
```

独立检测文件需有 `TargetID`，顺序与 beta 文件完全相同；其样本列可用 `<Sentrix_ID>` 或 `<Sentrix_ID>.Detection.Pval`。
检测 p 值缺失时默认停止，不能声称完成了该项 QC。如果确认只有 beta 数据，可以显式加 `--allow-missing-detection-p`；这属于不同 QC 方案，会被记录。
仅检查文件、身份和标签，不训练：

```bash
bash run.sh --data-dir "/path/to/COMET_RMS_Data" --preflight-only
```

## 默认执行的步骤

1. 读取包内固定划分 `inputs/cohort/patients.csv`；核对队友电脑上的原始身份和标签。不会重新随机切分。
2. 读取所有原始 CpG，做标签无关的检测 p / beta 范围 QC，保留完整 probe universe。患者缺失率 >20% 会停止，不悄悄更换 holdout。
3. 在相同训练患者和相同外层 folds 上比较两条 Ridge 路线：
   - **variance**：训练折内方差最高的 1,000 个 CpG；
   - **limma**：训练折内按 fusion POS/NEG 差异排名最高的 1,000 个 CpG。
4. 两者都是外层 5-fold、内层 3-fold。特征筛选、95% probe coverage、median imputation、scaling 均在各内层训练折独立拟合。C 用内层 BA 调参；每患者仅一个代表样本。
5. 用教师训练集的 nested OOF BA 比较方法；BA 相同看 OOF ROC AUC，完全相同优先 LIMMA。该比较是探索性的，选择后的 OOF 分数不能当无偏最终性能估计。
6. 在 108 位训练患者上拟合最终教师，评分两个方法的 validation；**默认不评分 test**。
7. 用选中的教师给 79 位允许训练的缺标签候选预测，保留全部分数和来源。初始纳入标准为 `max(p,1-p) >= .90`，不是保证 90% 正确率；gold=1、silver=.3 的 loss weights 是后续实验起点。
8. 导出学生训练名单、配对患者的已选 CpG beta、图片关联清单和结果包。

LIMMA 不保证比方差筛选更好，所以两条路线都保留。它是按目标标签找组间差异，不是寻找“组内变化最大”。
为隔离特征选择的影响，两种路线都选 1,000 个 CpG，Ridge 都输入标准化 beta；LIMMA 内部仅在差异分析时使用 M-values。
`--top-k 849` 可以改面板大小，但那是新配置，应使用新的 `--out`，不是复用旧面板。

## LIMMA 具体做什么

代码在 `src/ridge_teacher/features/limma_select.R`，调用真正的 R limma，不是用普通 t-test 冒充。

- 仅根据当前训练折做 probe coverage 筛选、非恒定检查和训练中位数填补。
- 将 beta 截到 `[1e-6, 1-1e-6]`，计算 `M = log2(beta / (1-beta))`。
- 对每个 CpG 拟合 `M ~ fusion_status`，NEG=0、POS=1。
- `lmFit` + `eBayes` 得到 moderated t 和双侧 p 值。
- 在全部受检 CpG 上做 BH 调整，再按原始 p 值排序选 top-k，平局按原始 feature index。
- 最终保存 CpG ID、delta_M、moderated_t、p、BH-adjusted p、受检 CpG 数和训练人数。
- 没有强制 FDR<.05 门槛：top-k 是预测特征面板，不表示全部被证明是显著生物标志物。

官方文档：[limma](https://bioconductor.org/packages/limma)。

## 输出在哪里、发回什么

默认输出在 `outputs/teacher_v1/`：

```text
results_summary.csv                       # 两种方法的 OOF / validation 指标
teacher_selection.json                    # 如何选择生成 silver 的教师
variance/teacher/                         # 模型、OOF 预测、CpG IDs
limma/teacher/                            # 同上，另有 LIMMA 统计表
variance/validation/                      # 患者级预测及指标
limma/validation/
silver/silver_labels.csv                  # 所有候选的分数、预测、纳入标记和来源
silver/student_train.csv                  # gold 19 + 纳入的 silver，保留 label_source
silver/paired_selected_cpg_beta_imputed.csv # 138 位用于本轮训练/评估的配对患者的面板 beta；有 split 字段
silver/panel_provenance.json               # CpG 面板来源
silver/image_input_manifest.csv           # 下游影像关联清单
matrix/sample_qc.csv                      # 原始数值 QC 结果
pipeline.log
return_results.zip                       # 将这个私有结果包发回项目组
```

`return_results.zip` 包含模型、结果、已选 CpG 输入和日志，不包含完整原始矩阵或 CV 缓存。
`paired_selected_cpg_beta_imputed.csv` 包含 holdout 的 **features**，不是可直接混在一起训练的 CSV；必须遵守 split。
实际患者 ID / 标签 / CpG 数据属于私有输入输出，已加入 `.gitignore`，不应随代码提交公共仓库。

本包已经换成 v4 划分。若队友曾运行旧包，必须用全新的输出目录重新训练，不能复用旧模型或缓存。

训练中断后继续同一配置：

```bash
bash run.sh --data-dir "/path/to/COMET_RMS_Data" --resume
```

已完成步骤复用，中断的未完成步骤保留为 `.interrupted.*` 后重跑。原始文件、划分、代码、环境或参数改变时不允许混用旧结果。
确定方法和阈值后，才单独打开 test 评估：

```bash
bash run.sh --data-dir "/path/to/COMET_RMS_Data" --resume --include-test
```

默认两个选择方法会各有一行 test 结果；不使用 test 再挑方法。只跑 LIMMA 可用 `--selector limma --out outputs/limma_only_v1`。

## 代码位置

```text
run.sh / environment.yml       一键环境与运行入口
run.py                         简短 Python 入口
src/ridge_teacher/
  cohort/split.py              审核患者、选代表样本、train/validation/test 划分
  cohort/policy.py             仅 ERMS / ARMS 可以进入任何训练集
  data/preflight.py            核对原始患者身份和 fusion 标签
  data/cache.py                原始矩阵、检测 p、样本 QC、磁盘缓存
  data/contracts.py            防止数组错位、重复患者、holdout 混入训练
  features/selection.py        方差选择器和 LIMMA 的 sklearn 接口
  features/limma_select.R      M-value、线性模型、经验贝叶斯、p / BH
  train/search.py             训练折内 Pipeline 和内层 C 搜索
  train/fit.py                外层 OOF、最终 Ridge 拟合、模型保存
  inference/predict.py        ridge teacher predict / silver 导出
  evaluation/metrics.py       患者级 BA、ROC AUC、PR-AUC(AP)、recall
  image_processing/manifest.py 关联 SVS，导出下游影像输入清单
  runner.py                   步骤编排、恢复执行、选择教师、结果打包
inputs/cohort/                 私有固定划分
inputs/reviewed/               私有最小审核来源和人数差异
outputs/                      私有运行产物
```

`prepare.py / cache_matrix.py / teacher.py` 仅保留旧命令兼容入口。
整个 MedGemma 仓库原有 GPU / FL 入口没有被无关迁移，避免破坏现有训练命令。
真正的 WSI tissue segmentation、patch extraction、H-Optimus embedding、ABMIL 仍在仓库的 `data_preprocessing/`，本包不重复实现它们。
`image_processing/manifest.py` 只导出关联，不代表图像已处理好。

重建划分（一般不要这样做；已有冻结版本）：

```bash
CAPSTONE_USE_EXISTING_ENV=1 .env/bin/python prepare.py \
  --samples inputs/reviewed/samples_minimal.csv \
  --inventory inputs/reviewed/image_inventory.json \
  --out outputs/rebuilt_cohort
```

## 交接前已验证的范围

```bash
bash run.sh --self-test
```

包含患者隔离、重复/冲突标签、嵌套预处理、真正的 LIMMA、两种 selector 的完整合成数据运行、环境/路径空格、恢复执行、显式 test 开关等测试。
macOS Apple Silicon 上已经实测环境创建和完整流程；其他系统使用同一 environment.yml，尚未逐台实测。

还需要队友原始数据验证：矩阵是否覆盖本轮227个代表数组、检测p值、样本QC和真实模型性能。若失败，先保留并审核错误，不要跳过病例凑够20+20。
当前是探索性重新划分，一些病例在历史实验中已被查看，不能称全新的外部盲测。

后续多模态训练必须按新split重新训练监督式ABMIL/选图和相关面板，不能复用看过新holdout的旧checkpoint。
`student_train.csv` 是数据契约；后续MedGemma训练器还需接入图像和loss weights，不是本包自动启动的GPU训练。
