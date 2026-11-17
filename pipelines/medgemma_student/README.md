# Gold + silver MedGemma student (v4)

这条流水线将已完成的 Ridge 返回包连接到新划分下的 ABMIL、spatial16 和集中式 MedGemma DDP。
**代码已接通；请按 [VM_DNANEXUS.md](VM_DNANEXUS.md) 在VM上从DNAnexus下载WSI并生成输入，再开始GPU短跑。**

## 固定实验定义

- 训练：91 位，19 gold（10 POS / 9 NEG）+ 72 silver（22 POS / 50 NEG）。
- Validation：20 位真实标签，12 POS / 8 NEG；test：20 位真实标签，12 POS / 8 NEG。
- 所有训练患者仅限 ERMS / ARMS；Spindle / RMS-NOS 的既定 holdout 不移入训练。
- CpG：返回包中所选 LIMMA teacher 的同一组 1,000 个 beta 值；不重新使用 holdout 选特征。
- H-Optimus-1 冻结，仅复用无监督/预训练编码的特征，不复用旧 ABMIL 权重或旧 attention。
- ABMIL：新建 gated ABMIL，使用这 91 位训练患者，gold=1.0 / silver=0.3，固定 50 epochs、Adam LR=1e-4、weight decay=1e-5、seed=20261117；每张训练 WSI 最多固定抽取 4,096 tiles。最终对每张WSI已编码的候选tiles打attention分数；候选池上限4,000，与旧缓存一致。
- ABMIL 不计算 holdout 指标、不根据 validation 选择 epoch；holdout 标签不传入训练步骤。
- Spatial16：按 attention 降序遍历全部已编码候选 tiles，tile 原点的欧氏距离至少为 `2 × tile_level0`。这是原点距离规则，不是两张 tile 之间留出两个完整空白 tile。**禁止距离放宽或不足16张时填充重复图。** 若不能选满16张，失败并人工审查。
- 为复用原来的输入加载器，将16张224×224 tiles无损存入896×896 PNG容器。训练加载器会裁开，真正传给 MedGemma 的是 **16 张独立图片 + 一份 CpG 文本 + 一个患者回答**，不是一张 montage。
- Student loss：`mean_patients(weight × mean_answer_tokens(CE))`。gold权重1.0，silver权重0.3。只监督回答，prompt/image tokens仍被mask；**不除以当前batch的权重之和**，否则batch=1时silver权重会失效。
- 本次加权接口用于 `train/centralized.py`。没有把新的silver数据默认为已经支持所有联邦算法。

## 当前缺口（2026-11-17 本地检查）

共需要131位患者的影像：现有H-Optimus备份覆盖48位（train14 / validation17 / test17），还缺83位特征（train77 / validation3 / test3）。Capstone内没有这131张原始WSI。旧16-tile图片不能替代新选图所需的原始WSI。

清单保存在：`../outputs/medgemma_student_v4_20261117/plan/asset_inventory.csv` 和 `missing_assets.csv`（从仓库根目录看）。`missing_assets.csv`包含全体131行，以两个missing字段区分缺失项。

## 可选：已有完整本地WSI时手动准备

从 `medgemma-fl` 根目录运行。WSI按 `slide_id.svs` 命名；不要重命名不相关切片冒充缺失样本。
需要现有训练环境及 `requirements-wsi.txt`；CPU数据处理也需PyTorch和系统OpenSlide。H-Optimus-1模型需要账号有访问权。冻结编码器生成83张新WSI特征可能很慢；CPU能运行，但没有实测耗时，不承诺在笔记本短时间完成。

```bash
cd /path/to/medgemma-fl
# 改成实际原始WSI目录，可放在外置盘；这里不自动下载或推断来源。
WSI_DIR='/absolute/path/to/raw_wsi'
OLD_FEATURES='../vm_backup/final_experiments_20261108/snapshots/montage-new/data/wsi/hoptimus1'
WORK='../outputs/medgemma_student_v4_ready'

python -m pipelines.medgemma_student plan \
  --returned ../return_results \
  --cohort ../outputs/silver_labels_20261117/cohort_v4 \
  --feature-dir "$OLD_FEATURES" --slide-dir "$WSI_DIR" \
  --output "$WORK/initial_plan"

# 仅补提缺失特征。若已有完整H-Optimus特征，跳过这一步，下一步改成其目录。
python -m pipelines.medgemma_student embed-missing \
  --plan "$WORK/initial_plan/plan.json" --output "$WORK/new_features" --device cpu

# 汇入新特征后生成新计划；旧计划保留用于审计。
python -m pipelines.medgemma_student plan \
  --returned ../return_results \
  --cohort ../outputs/silver_labels_20261117/cohort_v4 \
  --feature-dir "$WORK/new_features" --feature-dir "$OLD_FEATURES" \
  --slide-dir "$WSI_DIR" --output "$WORK/complete_plan"

python -m pipelines.medgemma_student fit-selector \
  --plan "$WORK/complete_plan/plan.json" --output "$WORK/abmil" --device cpu
python -m pipelines.medgemma_student extract \
  --plan "$WORK/complete_plan/plan.json" --selector "$WORK/abmil" --output "$WORK/tiles"
python -m pipelines.medgemma_student export \
  --plan "$WORK/complete_plan/plan.json" --tiles "$WORK/tiles" --output "$WORK/medgemma_inputs"
python -m pipelines.medgemma_student verify --data "$WORK/medgemma_inputs"
```

每个输出目录必须是新的，防止把旧划分或失败任务产物混进来。`plan`不会删除缺失患者；`fit-selector`发现缺失features会在训练前停止；`extract`缺WSI或不满足16张间距时停止。只有`export`完成并通过`verify`，`medgemma_inputs/`才是可训练数据。

CPU ABMIL可运行；固定配置是本轮预先指定的起点，尚未以真实91位数据执行。若提特征最终需要GPU，先准备好WSI和下载，再租GPU，不能把“代码已完成”当成“图像已生成”。

## VM环境和上传

将**这次更新后的仓库**和完整的 `medgemma_inputs/` 上传到VM；后者是自包含目录，图片已复制，不依赖本地绝对路径或软链接。无需上传12GB原始beta矩阵或所有WSI来训练MedGemma。源码使用 GitHub 的 `comet-fusion-pos-neg` 分支；患者输入、特征和模型文件通过私有运行包单独传输，不放入 Git。

使用之前验证过的CUDA训练环境；依赖版本记录在 `requirements-vm.txt`。新Linux VM可按该文件创建环境：

```bash
cd /home/ubuntu/medgemma-fl  # 替换为实际仓库路径
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.13.0+cu126 torchvision==0.28.0+cu126 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-vm.txt
python -m pip install -e . --no-deps
# 确保HF账号已获得 google/medgemma-1.5-4b-it 权限，并通过已有安全登录方式认证。
```

版本文件是之前VM的环境记录；当前没有新的GPU机器，因此本轮CUDA/TRL端到端短跑仍需在VM上验证。不要把两次benchmark同时运行。

## 单卡和四卡测试命令

在同一台四卡VM上先单卡，再四卡。两次均从相同基础模型重新初始化，使用相同数据、seed、LoRA范围、LR和有效batch=4；默认4-bit QLoRA、BF16计算、SDPA、非重入activation checkpointing。不得拿单卡FP32与四卡QLoRA比速度。

```bash
# 已进入仓库并激活环境；输入路径可自行替换。
bash scripts/benchmark_silver_student.sh 1 data/silver_v4 runs/silver_v4_bench_1gpu
bash scripts/benchmark_silver_student.sh 4 data/silver_v4 runs/silver_v4_bench_4gpu
python scripts/compare_student_benchmarks.py runs/silver_v4_bench_1gpu runs/silver_v4_bench_4gpu
```

- 默认 `lora-only` = vision + language LoRA；可加第4个参数 `language-only` 做另一个受控比较。
- 30个optimizer steps，前5步预热不计入稳态均值。benchmark先做真实processor检查，再启动模型训练。
- 单卡每卡batch=1、累积4；四卡每卡batch=1、累积1。一次训练的总有效batch都为4。
- 记录每rank的峰值显存、训练总时长、step耗时和总GPU-hours；比较脚本校验输入hash及关键设置。
- DDP会填充/重复少量样本以对齐各rank；91不能被4整除，四卡每epoch可能处理92个记录。不能声称单/四卡优化轨迹完全一样。
- 4卡DDP复制完整模型，不能合并显存；真实OOM/吞吐未知。benchmark checkpoint只是短跑结果，不作为科学结论。
- 本次没有更改阈值、启动test评估或使用test挑选设置。

## 短跑通过后的正式训练示例

先根据GPU短跑决定使用1卡还是4卡；下面是4卡示例，沿用60 epochs及2/5/10/20/30/50/60保存点。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train/centralized.py \
  --data-root data/silver_v4 --output runs/silver_v4_60epochs \
  --adapter_scope lora-only --weighted_labels \
  --per_device_train_batch_size 1 --effective_batch_size 4 \
  --num_train_epochs 60 --save_adapter_epochs 2 5 10 20 30 50 60 \
  --learning_rate 5e-5 --seed 20261117 --max_seq_length 32768 \
  --attn_implementation sdpa --no-local_eval
```

`--no-local_eval`仅关闭训练中的validation loss；后续需另用候选答案评分评估已保存checkpoint的BA/AUC，不能把训练loss当成分类结果。本示例是前台命令，需保持会话或自行使用tmux/nohup，不能直接合盖假设它继续。

## 已执行的检查

33项CPU检查通过，覆盖新数据导出/权重、holdout标签不影响ABMIL拟合、严格间距、两个真实Gloo进程的梯度一致性、真实TRL/Transformers配合小型Gemma3的梯度累积（包括不足4个样本的最后一组），以及既有多图输入回归；还覆盖DX下载完整性、断点续传、错误清单拒绝及独立GPU任务分配。测试未下载MedGemma权重、未执行CUDA训练；不等于四张GPU已验证可用。
