# LoRA-only：两套 montage × 五组训练的 VM 操作手册

这次固定 10 个正式实验，均为 5 rounds × 2 local epochs、LoRA rank 16、学习率 2e-4、batch 1、gradient accumulation 4。每套 montage 包括 centralized、IID FedAvg、IID FedLC、skew08 FedAvg、skew08 FedLC。保留已有患者划分与 849-CpG 输入，只改变完整 embedding/head 是否训练，以及选图规则。训练仍使用 4-bit 基础模型。

本地代码与输入检查通过；新配置还需要 GPU smoke test。不要从旧 MedGemma checkpoint 继续训练，要从同一个基础模型重新开始。旧数据、ABMIL、特征及基线结果保留。

## 1. VM 选择与第一步检查

最稳妥是复用上次成功运行的 A100 80GB 机器镜像。备份 gpu.txt 记录 A100 80GB、驱动 570.195.03，实际 torch 是 2.11.0+cu128、torchvision 0.26.0+cu128。优先沿用该环境；不要直接按仓库 requirements 的 2.13.0 重装。

新机采用 Linux x86_64、NVIDIA 驱动已安装、Python 3.12 可用的镜像。下面统一使用 `~/medgemma-fl`，适用于 shadeform 或 ubuntu 用户。先查看挂载磁盘：

```bash
nvidia-smi
python3 --version
uname -m
lsblk
df -h
```

本手册安装环境以 Python 3.12 为起点。若镜像没有相应 Python，先选匹配镜像或完成 Python 安装，不要把系统 Python 原地升级。已有成功环境可跳过新建 venv 和重装依赖。

一台 GPU 可以顺序完成十组；想并行时，建议先准备一次输入，再复制到两台相同配置的 VM，分别跑 top16 和 spatial16。每个 FL 实验仍在一台 VM 模拟三个 client，不需要每个 client 单独租 VM。

## 2. VM：获取代码，建立目录

新 VM：

```bash
sudo apt-get update
sudo apt-get install -y git rsync tmux build-essential python3-dev python3-venv \
  openslide-tools libopenslide0 libgl1 libglib2.0-0 libgomp1
cd ~
git clone --branch comet-fusion-pos-neg --single-branch https://github.com/QianyuFan8/medgemma-fl.git
cd ~/medgemma-fl
mkdir -p logs data runs/fixed849_disjoint
```

已有 checkout 则执行：

```bash
cd ~/medgemma-fl
git status --short
git pull --ff-only origin comet-fusion-pos-neg
```

若提示本地代码冲突，先保留并检查改动；不要使用 reset --hard。代码应包含 scripts/run_fusion_suite.sh 与 scripts/prepare_spatial_inputs.sh。

## 3. Mac：恢复备份输入，不上传旧 MedGemma 权重

替换 SSH 用户、IP 和私钥。下列命令在 Mac 终端运行：

```bash
COMET_VM='shadeform@VM_IP'
COMET_KEY="$HOME/.ssh/YOUR_KEY.pem"
COMET_BACKUP='/Users/wuzhentian/Desktop/Harvard_Fall/Capstone/vm_backup/fixed849_20261028'
rsync -av -e "ssh -i $COMET_KEY" "$COMET_BACKUP/data/" "$COMET_VM:~/medgemma-fl/data/"
rsync -av -e "ssh -i $COMET_KEY" "$COMET_BACKUP/runs/fixed849_disjoint/abmil/" \
  "$COMET_VM:~/medgemma-fl/runs/fixed849_disjoint/abmil/"
rsync -av -e "ssh -i $COMET_KEY" "$COMET_BACKUP/logs/environment-pip.txt" \
  "$COMET_VM:~/medgemma-fl/logs/previous-environment-pip.txt"
```

备份 data 约 1.7GB，包含 paired、central、iid、skew08、旧 montage 和 60 个 H5 特征文件。无需上传完整 runs 下的旧大型模型 checkpoint。无需重新生成患者 split，不要重跑 limma 或 H-Optimus/ABMIL 训练。

## 4. VM：恢复依赖并登录 Hugging Face

先在 `df -h` 中确认大数据盘。上次 Shadeform 使用 `/ephemeral`；如果新机不同，将以下 COMET_DISK 换成实际挂载点。ephemeral 不保证关机/销毁后的保存，实验结果必须另行备份。不要把约 90GB 的 SVS 放进上次仅 100GB 的系统盘。

```bash
COMET_DISK=/ephemeral
sudo mkdir -p "$COMET_DISK/comet_slides" "$COMET_DISK/hf"
sudo chown "$USER:$(id -gn)" "$COMET_DISK/comet_slides" "$COMET_DISK/hf"
export HF_HOME="$COMET_DISK/hf"
cd ~/medgemma-fl
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128
python - <<'PY'
from pathlib import Path
lines=Path('requirements.txt').read_text().splitlines()
Path('logs/requirements-without-torch.txt').write_text('\n'.join(
    line for line in lines if not line.startswith(('torch==','torchvision=='))) + '\n')
PY
python -m pip install -c logs/previous-environment-pip.txt \
  -r logs/requirements-without-torch.txt -r requirements-wsi.txt pandas dxpy
python -m pip check
python - <<'PY'
import torch, torchvision, transformers, peft, trl, nvflare, openslide, h5py
print('Torch/CUDA:',torch.__version__,torch.version.cuda)
print('Transformers/PEFT/TRL:',transformers.__version__,peft.__version__,trl.__version__)
assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0))
PY
hf auth login
python -m pip freeze > logs/ablation-environment-pip.txt
git rev-parse HEAD > logs/ablation-code-commit.txt
```

依赖约束引用上次真实运行记录，而不是安装全部历史环境的无关包；其中应包括 transformers 5.21.0、peft 0.21.2、trl 1.16.0、nvflare 2.7.3。若新 Python/驱动无法安装这些版本，先解决环境兼容性，不要静默放宽所有版本。PyTorch CUDA 12.8 wheel 对应命令来自 https://pytorch.org/get-started/previous-versions/ 。

HF_HOME 需要在新 SSH/tmux 会话中重新 export；不要记录 token。MedGemma 模型访问权限沿用你的 Hugging Face 账号。

## 5. 验证原 montage 输入；它已经可以训练

```bash
cd ~/medgemma-fl
source .venv/bin/activate
export COMET_CONTEXT=32768
python scripts/check_fusion_splits.py --prepared data/fixed849_disjoint_paired \
  --export data/fixed849_disjoint_central --export data/fixed849_disjoint_iid \
  --export data/fixed849_disjoint_skew08
python check_methylation_context.py --data-dir data/fixed849_disjoint_central \
  --max-seq-length "$COMET_CONTEXT"
```

应报告 60 条记录、跨 train/validation/test 患者互斥。这里不等同于历史 CpG 筛选完全独立，也不等同于每条记录都是不同患者。

## 6. 准备空间分散 montage（需要 SVS，仅执行一次）

若原机器仍有 SVS，直接设置 COMET_SLIDE_DIR 指向它们。新 VM 需重新下载；备份没有 SVS。以下延续第 4 步的大盘变量：

```bash
export COMET_SLIDE_DIR="$COMET_DISK/comet_slides"
dx login
dx select project-JBXfX9Q0vQQYJ9k4BxGVyYzJ
python - <<'PY'
import csv
from pathlib import Path
with Path('data/fixed849_disjoint_paired/splits.tsv').open() as f:
    names=sorted({r['slide_id']+'.svs' for r in csv.DictReader(f,delimiter='\t')})
Path('logs/required-svs.txt').write_text('\n'.join(names)+'\n')
print('Required SVS:',len(names))
PY
set -euo pipefail
while IFS= read -r name; do
  if [[ ! -s "$COMET_SLIDE_DIR/$name" ]]; then
    dx download "project-JBXfX9Q0vQQYJ9k4BxGVyYzJ:/COMET_RMS/$name" \
      -o "$COMET_SLIDE_DIR/${name}.part"
    mv "$COMET_SLIDE_DIR/${name}.part" "$COMET_SLIDE_DIR/$name"
  fi
done < logs/required-svs.txt
bash scripts/prepare_spatial_inputs.sh
python check_methylation_context.py --data-dir data/fixed849_spatial16_central \
  --max-seq-length "$COMET_CONTEXT"
```

若下载中断留下 .part，先检查并处理该临时文件后再重试。脚本会用原 ABMIL attention，从前 128 个候选中选间距至少两个 tile 宽度的 16 个；生成新 montage，导出 central、iid、skew08 三种输入，并核对新旧 beta、split、client assignment 完全一致。prompt 会自动描述选图规则。

已有输出目录会拒绝覆盖。如果导出中途失败，先定位原因并保留失败目录，不要把半成品当作完整输入继续训练。

## 7. 每套输入跑五组：先 smoke，再正式训练

下面可在一台 VM 顺序执行。两台 VM 并行时，在准备完成后将数据同步至第二台：A 只执行 top16 部分，B 只执行 spatial16 部分，两台使用相同代码和环境。训练阶段无需传 SVS，只需要各自导出的 montage/JSON/CSV 和 paired manifest。

```bash
tmux new -s fusion-ablation
cd ~/medgemma-fl
source .venv/bin/activate
export HF_HOME=/ephemeral/hf  # 换成第 4 步实际使用的路径
export COMET_CONTEXT=32768 COMET_SPLIT=validation
set -euo pipefail

# 原 top-attention montage：5 个 smoke job、5 个正式训练、5 个 validation 评价
bash scripts/run_fusion_suite.sh top16 smoke
bash scripts/run_fusion_suite.sh top16 train
bash scripts/run_fusion_suite.sh top16 eval

# 空间分散 montage：另外 5 个正式实验
bash scripts/run_fusion_suite.sh spatial16 smoke
bash scripts/run_fusion_suite.sh spatial16 train
bash scripts/run_fusion_suite.sh spatial16 eval
```

每个 smoke 只跑 1 round × 1 optimizer step，输出与正式训练分开。检查日志、GPU 显存及 trainable_parameters.json，确认 adapter_scope 为 lora-only，无完整 embedding/head 可训练参数。正式 suite 固定 5×2；FL 用三个逻辑 client、串行 GPU 调度。命令遇到失败就停止，不应忽略错误继续十组。

输出分别位于：

- runs/ablation_lora-only_top16_r5_e2/
- runs/ablation_lora-only_spatial16_r5_e2/

每个目录内包含 central、fl_iid_fedavg、fl_iid_fedlc、fl_skew08_fedavg、fl_skew08_fedlc 各自的 smoke/train/validation 输出。禁止在同一输出位置覆盖重跑。

若 suite 在中间失败，已完成的模型不要重跑；按 FUSION_ABLATIONS.md 的单模型 launcher 继续剩余组。默认变量 COMET_DATA_PREFIX 和 COMET_INPUT_TAG 要对应 top16 或 spatial16。

## 8. Test 与结果备份

十组设置固定后，再统一报告 test。现有 test 已参与开发讨论，后续结果应注明探索性；不要用 test 选择更多 epochs、阈值或选图超参数。

```bash
export COMET_SPLIT=test
bash scripts/run_fusion_suite.sh top16 eval
bash scripts/run_fusion_suite.sh spatial16 eval
```

保留逐条预测、evaluation.json、运行日志、trainable_parameters.json、selection_audit.json、环境 freeze 和 git commit。Mac 拉回结果时使用一个新的本地目录：

```bash
# 在 Mac 终端运行；COMET_VM/COMET_KEY 使用第 3 步的设置
COMET_RESULTS='/Users/wuzhentian/Desktop/Harvard_Fall/Capstone/vm_backup/lora_only_ablation_20261031'
mkdir -p "$COMET_RESULTS"
rsync -av -e "ssh -i $COMET_KEY" "$COMET_VM:~/medgemma-fl/runs/ablation_lora-only_top16_r5_e2/" "$COMET_RESULTS/top16/"
rsync -av -e "ssh -i $COMET_KEY" "$COMET_VM:~/medgemma-fl/runs/ablation_lora-only_spatial16_r5_e2/" "$COMET_RESULTS/spatial16/"
rsync -av -e "ssh -i $COMET_KEY" "$COMET_VM:~/medgemma-fl/data/wsi/fixed849_spatial16_montages/" "$COMET_RESULTS/spatial_montages/"
rsync -av -e "ssh -i $COMET_KEY" "$COMET_VM:~/medgemma-fl/logs/" "$COMET_RESULTS/setup_logs/"
```

确认结果已下载到持久存储后再销毁 VM。不要假定 ephemeral 数据会保留。
