# 代码目录与运行入口

## 三个概念：入口、实现、流水线

- **入口**：在终端执行的短命令，主要在 `train/`、`inference/`、`data_preprocessing/`。
- **实现**：训练循环、模型加载、loss、DDP等代码，在 `src/medgemma_fl/`。入口调用它们，不复制实现。
- **流水线**：按顺序连接多个阶段，在 `pipelines/`；历史实验比较队列在 `experiments/`。

根目录不再放Python运行文件，只保留说明、依赖、安装配置、LICENSE和各功能目录。

## 每个目录的作用

| 目录 | 内容与用途 |
|---|---|
| `train/` | MedGemma集中式训练、联邦任务创建、客户端训练入口。日常训练从这里开始。 |
| `inference/` | POS/NEG候选评分、checkpoint评估、阈值拟合/应用、结果汇总。 |
| `src/medgemma_fl/` | 实际实现和共享Python包；想理解训练代码时看这里。 |
| `data_preprocessing/` | 组织检测/切块、H-Optimus特征、ABMIL，以及cohort和输入导出工具。 |
| `pipelines/silver_labels/` | 新划分、训练亚型白名单、折内CpG选择、Ridge teacher和silver label；队友运行包对应的独立子项目。 |
| `pipelines/medgemma_student/` | 返回包核对、DX下载、新ABMIL选图、spatial16与gold/silver加权输入，是本轮主要流水线。 |
| `experiments/` | zero-shot、epoch、montage/多图、16/20图、联邦等历史实验队列入口；复现实验时使用。 |
| `scripts/` | Shell启动器、单卡/四卡短跑、检查和运行包构建工具。 |
| `tests/` | 患者隔离、输入、loss、DDP和任务导出的回归测试。 |
| `docs/` | 使用说明、性能说明、旧→新入口映射。 |
| `methylation/` | 早期R脚本与配置，为复现旧实验保留；新teacher优先用silver_labels。 |
| `UMAP/` | 早期UMAP分析代码；结果仅保留本地。属于EDA，不是监督式CpG选择。 |
| `assets/` | 文档插图，不参与训练。 |
| `data/`、`runs/`、`logs/` | 本地输入、训练结果和日志，Git忽略这些目录；本地有、GitHub没有是正常的。 |

外层 `Capstone/outputs/`、`return_results/`、`vm_backup/` 和PPT属于项目资料，不是这个Git仓库的Python源码。

## MedGemma训练具体在哪里

| 功能 | 终端入口 | 实际实现 |
|---|---|---|
| 集中式单卡/DDP | `train/centralized.py` | `src/medgemma_fl/training/centralized.py` |
| 联邦任务与server配置 | `train/federated.py` | `src/medgemma_fl/training/federated.py` |
| 客户端内部训练/DDP | `train/client.py`，通常由NVFlare调用 | `src/medgemma_fl/training/client.py` |
| 候选评分与评估 | `inference/medgemma.py` | `src/medgemma_fl/evaluation/evaluate_subtype.py` |
| 已保存联邦round评估 | `inference/federated.py` | `src/medgemma_fl/evaluation/evaluate_federated_tiles16.py` |
| 阈值拟合/应用 | `inference/threshold.py` | `src/medgemma_fl/evaluation/fusion_threshold.py` |

在 `src/medgemma_fl/` 中：

- `model.py`：基础模型和PEFT加载；`adapter_scope.py`：LoRA范围。
- `weighted_sft.py`：gold/silver加权回答loss。
- `distributed_training.py`：客户端内部DDP通信；`client_gpu_lock.py`：共享GPU的客户端互斥。
- `fl_methods.py`：各联邦方法的本地训练修正；`custom_aggregators.py`：server聚合。
- `job_packaging.py`：把包内代码带进NVFlare导出任务，防止远端客户端缺模块。
- `training_performance.py`：GPU分配与有效batch；`training_benchmark.py`：耗时记录；`training_cache.py`：预处理缓存。
- `preprocessing/`、`experiments/`：原来根目录的数据和实验实现，保留原文件名方便追溯。

不要把历史队列的默认epoch或LoRA范围当成本轮配置。

## CpG选择、Ridge训练和预测在哪里

Ridge handoff可以单独发给队友，因此保留自己的`src/`和环境文件。它与顶层MedGemma训练不是重复实现。

```text
pipelines/silver_labels/
  run.sh / run.py                     一步运行入口
  environment.yml / requirements.txt Python/R环境
  src/ridge_teacher/
    cohort/                          患者划分、ERMS/ARMS训练白名单
    data/                            原始矩阵读取、缓存与检查
    features/selection.py            折内过滤、填补、方差/limma选择
    features/limma_select.R           limma的R实现
    train/search.py                  CV与超参数搜索
    train/fit.py                     teacher拟合
    inference/predict.py             预测、置信度筛选与silver输出
    evaluation/                      指标计算
    image_processing/                影像关联清单，不是WSI编码器
```

以前说的`train/`只存在于这里的`src/ridge_teacher/train/`，并不代表仓库根目录已经整理好了。本次新增的顶层`train/`是MedGemma入口。

## 本轮建议阅读顺序

1. `pipelines/medgemma_student/plan.py`：核对患者、标签来源、CpG和影像。
2. `dnanexus.py`：云端文件清单、DX下载与完整性检查。
3. `embeddings.py` / `embedding_worker.py`：多GPU分工提取冻结H-Optimus特征。
4. `selector.py`：仅用训练集拟合新ABMIL，再严格选择16张spatial tiles。
5. `inputs.py`：导出16张独立图片、CpG文本与标签权重。
6. `scripts/benchmark_silver_student.sh` → `train/centralized.py`：同有效batch的单卡/四卡短跑。
7. `src/medgemma_fl/weighted_sft.py`：回答token的逐患者loss，gold=1.0、silver=0.3。

本轮加权student先用于集中式训练；目录整理不代表它已经自动接入每一种联邦算法。

## 为什么GitHub和本地不一样

2026-11-17核对时，本地分支为`comet-fusion-pos-neg`，本地HEAD与GitHub同名分支均为`67c81e9`。新代码还没有提交，因此GitHub没有这些本地改动。如果GitHub页面选择了`main`，还会看到另一条分支。

此前`.gitignore`忽略所有Markdown，新的说明也不能正常提交；本次取消这个全局规则，保留明确标为本地的旧说明例外。

**本地修改、commit和push是三个步骤。** GitHub 的 `comet-fusion-pos-neg` 分支用于同步经过检查的源码。新VM包另行提供所需患者输入和特征；这些资料、模型权重和运行结果不进入 Git。

## 安装与命令迁移

在已激活的环境、仓库根目录中执行：

```bash
python -m pip install -e . --no-deps
python train/centralized.py --help
python train/federated.py --help
python inference/medgemma.py --help
```

Editable install注册本地`medgemma_fl`包，使入口和子进程能导入代码。它不下载模型、不启动训练；`--no-deps`保留现有CUDA/PyTorch环境。

旧文件路径已迁移，详见`docs/entrypoint_migration.json`。旧复制命令可以改成`python scripts/legacy_entrypoint.py 原文件名.py ...`，由兼容入口调用同一份实现。正式命令优先使用新的`train/`、`inference/`路径。
