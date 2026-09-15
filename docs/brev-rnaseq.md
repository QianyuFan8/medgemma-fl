# Run the TARGET RNA-seq FL study on NVIDIA Brev

Use this runbook on a CUDA VM. A Mac cannot train MedGemma. Task design, data layout, and archived 3-round numbers live in the [README](../README.md). This file is only **login → copy → train → copy back → stop**.

Do not put Hugging Face tokens, Brev API keys, or raw `RNAseq/` matrices in git.

## 0. What you need

- Brev CLI on the laptop, logged into the org that can create GPUs.
- Hugging Face access to [`google/medgemma-4b-it`](https://huggingface.co/google/medgemma-4b-it) (same account on the VM).
- Local repo plus labeled counts: `RNAseq/TARGET_meta.txt` and `RNAseq/WU-TARGET-Batch{01–09}-*_RSEM_gene_count*.txt`. Skip `*TPM*` and unlabeled `WU-TARGET_AML_B*`.
- **Three GPUs, ≥40 GB each**, compute capability ≥ 8.0 (`bfloat16`). One A100 80GB is **not** enough for three simultaneous clients.

The working 3-site box was Crusoe **`l40s-48gb.4x`**: 4× L40S 48GB, stoppable, about **$6.96/hr** running and about **$0.03/hr** stopped (disk). Clients use GPUs 0, 1, 2.

## 1. Laptop — log in and create the instance

```bash
brev login
brev org ls          # switch org if needed: brev org set <name>
brev search --stoppable --gpu-name L40S --min-vram 40 --sort price
brev create medgemma-fl --type l40s-48gb.4x
brev ls
```

Wait until **STATUS=RUNNING** and **SHELL** is ready (not `BUILDING` / `NOT READY`). Creating a box starts billing.

If `l40s-48gb.4x` is out of capacity, pick another **stoppable** type with `gpu_count >= 3` and ≥40 GB per GPU. Non-stoppable 4× A6000 boxes are cheaper but you must finish in one sitting and `brev delete` (disk is gone).

## 2. Laptop — open a shell

```bash
brev shell medgemma-fl
nvidia-smi           # expect 4× L40S (or 3+ GPUs)
```

If SSH fails while BUILD is still running, wait and retry. `brev` will refresh SSH config.

Optional: `brev open medgemma-fl cursor` or `brev open medgemma-fl tmux`.

## 3. Laptop — copy code and labeled RNA-seq

From a **second** local terminal (keep the VM shell open):

```bash
cd /path/to/medgemma-fl

tar -czf /tmp/medgemma-fl-upload.tgz \
  --exclude='.venv' --exclude='.git' --exclude='__pycache__' \
  --exclude='RNAseq/WU-TARGET_AML*' --exclude='RNAseq/*TPM*' \
  --exclude='*_FL_global_model.pt' \
  *.py requirements.txt requirements-vm.txt README.md docs \
  data/rnaseq results \
  RNAseq/TARGET_meta.txt RNAseq/WU-TARGET-Batch*_RSEM_gene_count*.txt

brev copy /tmp/medgemma-fl-upload.tgz medgemma-fl:/home/ubuntu/medgemma-fl-upload.tgz
```

On the VM:

```bash
mkdir -p ~/medgemma-fl
tar -xzf ~/medgemma-fl-upload.tgz -C ~/medgemma-fl
cd ~/medgemma-fl
```

If `data/rnaseq` is already prepared on the laptop, you can skip raw counts and only copy `*.py`, `requirements-vm.txt`, and `data/rnaseq`.

## 4. VM — Python env (CUDA torch)

Do **not** `pip install -r requirements.txt` on the VM. That file pins CPU `torch==2.13.0` and conflicts with `torch==2.13.0+cu126`.

```bash
cd ~/medgemma-fl
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu126 -r requirements-vm.txt
```

## 5. VM — Hugging Face login

Accept the MedGemma license on the website first, then on the VM:

```bash
source .venv/bin/activate
hf auth login
```

Paste a token from that same account. `huggingface-cli` is deprecated; use `hf`. Export `HF_TOKEN` in **every new shell** (including a new `tmux` window) or eval will 401.

## 6. VM — preprocess (if `data/rnaseq` is missing)

```bash
source .venv/bin/activate
cd ~/medgemma-fl
python prepare_rnaseq.py --input_dir ./RNAseq --output_dir ./data/rnaseq
```

This writes `site-{1,2,3}/train.json`, `validation.json`, `eval.json`, and `split_label_distribution.svg`. Do not overwrite `./data/rnaseq` later when you build an IID split; use `--output_dir ./data/rnaseq-iid`.

## 7. VM — train in tmux

Three clients print tqdm on the same pane. Use tmux so an SSH drop does not kill the parent immediately. The FLARE simulator can still keep running after SSH dies; check `round_metrics.*.jsonl` before starting a second job.

```bash
tmux new -s fl
source ~/medgemma-fl/.venv/bin/activate
cd ~/medgemma-fl
```

**Archived baseline** (already in `results/baseline_noniid_3r1e/`; only rerun if you must):

```bash
python run_comm_study.py \
  --experiments lora-r4,lora-r8,lora-r16,lora-r32,qlora-r16 \
  --num_rounds 3 \
  --comm_log_root ./comm_logs \
  --gpu "[0],[1],[2]"
```

`run_comm_study.py` refuses to overwrite an existing `comm_logs/study_results.json` unless you pass `--overwrite`. Prefer a **new** log root.

**Next experiment** (class balance, 8 rounds, 2 local epochs, QLoRA r=16 only):

```bash
python run_comm_study.py \
  --experiments qlora-r16 \
  --name_suffix -balanced-8r2e \
  --num_rounds 8 \
  --num_train_epochs 2 \
  --balance_labels \
  --comm_log_root ./comm_logs_balanced_8r2e \
  --gpu "[0],[1],[2]"
```

Detach: `Ctrl-b` then `d`. Reattach: `tmux attach -t fl`.

Watch:

```bash
nvidia-smi
tail -f comm_logs_balanced_8r2e/qlora-r16-balanced-8r2e/round_metrics.site-1.jsonl
```

Job name / workspace is `/tmp/nvflare/simulation/<experiment_name>/`. `--name_suffix` avoids clobbering yesterday’s `qlora-r16` simulator dir.

## 8. VM — global eval

`run_comm_study.py` already evals when it finds `FL_global_model.pt`. To eval by hand:

```bash
source .venv/bin/activate
cd ~/medgemma-fl
python run_evaluation.py --task rnaseq \
  --eval_file ./data/rnaseq/eval.json \
  --tuned_model_path /tmp/nvflare/simulation/qlora-r16-balanced-8r2e/server/simulate_job/app_server/FL_global_model.pt \
  --finetune_only --max_samples 0
```

Majority-ALL floor on the archived eval set is **0.4364** (72/165).

## 9. Laptop — copy logs home

From the Mac:

```bash
brev copy medgemma-fl:/home/ubuntu/medgemma-fl/comm_logs_balanced_8r2e ./comm_logs_balanced_8r2e
```

Optional adapter (tens of MB, gitignored):

```bash
brev copy \
  medgemma-fl:/tmp/nvflare/simulation/qlora-r16-balanced-8r2e/server/simulate_job/app_server/FL_global_model.pt \
  ./qlora-r16-balanced-8r2e_FL_global_model.pt
```

Keep `results/baseline_noniid_3r1e/` untouched. Archive new JSON/SVG under a new `results/` folder when you are ready to commit.

## 10. Stop vs delete

```bash
brev stop medgemma-fl    # keeps disk (venv, HF cache, data); ~$0.03/hr on the 4× L40S box
brev start medgemma-fl   # later
brev delete medgemma-fl  # disk gone; only do this after logs are copied
```

`brev stop` only works on **stoppable** types (search FEATURES `S`). Idle **RUNNING** time is the expensive part.

## Common failures

| Symptom | What to do |
|---------|------------|
| `Could not resolve hostname medgemma-fl` | Wait for SHELL ready; `brev shell` retries with a refreshed SSH config. |
| pip / torch CPU vs `+cu126` | Install **only** `requirements-vm.txt` with the cu126 extra index. |
| Eval 401 | `HF_TOKEN` missing in this shell; `hf auth login` again. |
| SSH `exit 255` mid-job | Simulator may still be running. Check jsonl and `nvidia-smi` before restarting. |
| Garbled tqdm | Three clients share one tmux pane; ignore or split panes. |
| Global acc = 0.4364 | Majority ALL; expected for the short unbalanced 3×1 protocol. |
| Overwriting baseline | New `--comm_log_root` and `--name_suffix`; never `--overwrite` on `./comm_logs` if you still need those files. |

## After this non-IID rerun — IID (later)

Do not start IID until the balanced 8×2 run finishes and you have copied logs.

```bash
python prepare_rnaseq.py --split_strategy random --output_dir ./data/rnaseq-iid
python run_comm_study.py \
  --data_dir ./data/rnaseq-iid \
  --eval_file ./data/rnaseq/eval.json \
  --experiments qlora-r16 \
  --name_suffix -iid-balanced-8r2e \
  --num_rounds 8 --num_train_epochs 2 --balance_labels \
  --comm_log_root ./comm_logs_iid_balanced_8r2e \
  --gpu "[0],[1],[2]"
```

`--eval_file ./data/rnaseq/eval.json` keeps the same held-out patients as the non-IID baseline. Confirm that is the comparison you want before launching.
