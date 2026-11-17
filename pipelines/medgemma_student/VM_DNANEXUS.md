# DNAnexus → spatial16 → gold/silver student: VM runbook

Prepared 2026-11-17. Use the portable `medgemma_student_vm_20261117_v2.tar.gz` bundle; it contains the prepared source-code snapshot, frozen inputs, and 48 reusable H-Optimus-1 feature bags. It contains patient-linked research data; transfer only to your authorized research VM. No credentials are included.

## What is ready, and what still needs a VM

- 91 training patients: 19 gold + 72 accepted silver. Loss weights are 1.0 / 0.3, for both the new ABMIL selector and the MedGemma student.
- Validation and test each contain 20 gold-labeled patients; neither enters ABMIL or student fitting. Training is restricted to ERMS/ARMS.
- The returned LIMMA panel has 1,000 CpGs. No feature selection or teacher training is repeated here.
- Existing frozen features cover 48 of 131 required slides; 83 must be encoded. All 131 original WSI are needed to extract newly selected tiles, including the 48 slides whose features are cached.
- The Nov16 historical DNAnexus inventory contains all 131 filenames. The Oct26 size inventory estimates **226.9 GiB**. These are historical observations, not a fresh cloud verification. The local DX token has expired and the browser is logged out. The VM will authenticate and freeze current file IDs/sizes before downloading.
- Project: `project-JBXfX9Q0vQQYJ9k4BxGVyYzJ`, folder: `/COMET_RMS`.
- Use at least a **500 GB working disk**, preferably 32+ vCPUs and 128 GB RAM for four concurrent image workers. For the requested scaling test use four identical A100 GPUs (80 GB preferred; actual memory use still requires the short run). A one-GPU VM also supports preparation and the one-GPU test.
- No real 91-patient ABMIL run, CUDA MedGemma run, or four-GPU speed measurement has been completed in this preparation step.

## 1. Upload and unpack the bundle

Upload the `.tar.gz` and its `.sha256` file to the VM, then run in that directory:

```bash
sha256sum -c medgemma_student_vm_20261117_v2.tar.gz.sha256
tar -xzf medgemma_student_vm_20261117_v2.tar.gz
cd medgemma_student_vm/repo
```

This bundle includes a source-code snapshot. Source updates are maintained on the `comet-fusion-pos-neg` branch; Git does not include the private inputs or feature bags.

## 2. Install and log in

Ubuntu/Linux, Python 3.11 or 3.12, NVIDIA driver compatible with CUDA 12.6 wheels. Run these commands in the same shell. Replace `/mnt/student_v4` with an absolute path on the VM's large mounted disk; do not assume `/mnt` itself is a large disk.

```bash
sudo apt-get update
sudo apt-get install -y python3-venv libopenslide-dev openslide-tools tmux
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.13.0+cu126 torchvision==0.28.0+cu126 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-vm.txt -r requirements-wsi.txt dxpy==0.419.0
python -m pip install -e . --no-deps

export STUDENT_WORK=/mnt/student_v4
mkdir -p "$STUDENT_WORK"
df -h "$STUDENT_WORK"
export HF_HOME="$STUDENT_WORK/hf"
dx login
dx whoami
hf auth login
hf auth whoami
python scripts/verify_student_bundle.py ..
```

Your Hugging Face account needs access to both `bioptimus/H-optimus-1` and `google/medgemma-1.5-4b-it`. Log in after setting `HF_HOME`; do not paste tokens into scripts or send them back in results. The version file reflects the earlier VM environment, not a newly validated CUDA installation.

## 3. Prepare the data

Start a persistent terminal from the activated shell above:

```bash
tmux new -s student-prep
# Inside tmux, still in medgemma_student_vm/repo:
source .venv/bin/activate
export STUDENT_WORK=/mnt/student_v4
export HF_HOME="$STUDENT_WORK/hf"
set -o pipefail
bash scripts/prepare_student_vm.sh "$STUDENT_WORK" 0,1,2,3 2>&1 | tee "$STUDENT_WORK/prepare.log"
```

For a one-GPU VM replace `0,1,2,3` with `0`. Detach with Ctrl-B, then D; reconnect with `tmux attach -t student-prep`. The VM must remain running. Closing the laptop does not close tmux.

The script performs these stages sequentially:

1. Verify bundled file hashes, available GPU IDs, DX authentication and small model-config access.
2. Read the cloud inventory, require exactly one closed/live file per required filename, freeze immutable file IDs and sizes into `cloud_files.json`. No whole-project wildcard download.
3. Download the 131 files with DX, require remaining-download space plus 75 GiB reserve, verify byte sizes and write SHA-256 receipts. Interrupted downloads can resume; unverified existing final files cause an error.
4. Generate only the 83 missing H-Optimus feature bags. Four GPUs process disjoint slide lists with one frozen encoder per GPU. This is parallel feature extraction, not DDP training. Each WSI has at most 4,000 tissue candidates, matching the archived preprocessing command; seed=20261011, tile=224 px, target MPP=0.5, Bioptimus normalization. Tissue detection uses the existing HSV-saturation thumbnail mask (threshold 8, tissue fraction 0.5), not manually annotated ROI.
5. Fit a fresh gated ABMIL on the 91 training patients only: 50 fixed epochs, LR=1e-4, gold/silver weighting. No old supervised ABMIL weights are reused. Score the available candidate tiles of all 131 slides without using holdout labels.
6. Select 16 tiles per slide with origin-to-origin distance ≥2 tile widths; never relax spacing or duplicate tiles to fill a shortfall. This distance rule does not mean two empty tiles between tile edges. Failure to find 16 requires review.
7. Export and verify self-contained MedGemma inputs. The PNG container is split by the loader into 16 independent image inputs; it is not passed as one montage. CpG text appears once per patient.

Completion prints `READY_FOR_BENCHMARK: .../medgemma_inputs`. A log ending before this is not proof that preparation finished. Worker progress is in `new_features/worker-*.log`.

Re-running the same preparation command resumes downloads and feature extraction with identical configuration. Completed later stages are reused and provenance is checked downstream. Partial ABMIL/tiles/export directories deliberately stop rather than mix outputs; inspect the error before moving the incomplete directory aside and retrying. Do not delete source data or change the frozen split to bypass a failure.

The 48 cached bags were made under the archived 4,000-candidate configuration. Their H5 metadata records model, normalization, tile geometry and coordinates; it does not contain every historical preprocessing argument. The bundle includes the archived command as provenance. New features record explicit settings and source hashes.

## 4. Compare one GPU with four GPUs

Only after preparation prints `READY_FOR_BENCHMARK`, run these sequentially on the same four-GPU VM, inside tmux and the activated environment:

```bash
bash scripts/benchmark_silver_student.sh 1 "$STUDENT_WORK/medgemma_inputs" "$STUDENT_WORK/bench_1gpu"
bash scripts/benchmark_silver_student.sh 4 "$STUDENT_WORK/medgemma_inputs" "$STUDENT_WORK/bench_4gpu"
python scripts/compare_student_benchmarks.py "$STUDENT_WORK/bench_1gpu" "$STUDENT_WORK/bench_4gpu" | tee "$STUDENT_WORK/benchmark_comparison.json"
```

Both start from the same base model and run 30 optimizer steps, with 5 warmup steps excluded from steady-state timing. Same effective batch=4, LR=5e-5, seed, 16-image data, weighted loss, vision+language LoRA (`lora-only`), 4-bit loading, BF16, SDPA and activation checkpointing. One GPU accumulates four microbatches; four GPUs accumulate one per rank. DDP pads the 91-patient dataset, so trajectories are not claimed to be exactly identical. Activation checkpointing saves memory by recomputation; it is not itself a speed optimization.

These are training smoke/timing tests, not final model performance experiments. They do not run test-set scoring. DDP replicates the model on each GPU and does not pool GPU memory. The four-GPU gain must be measured, not assumed.

Output paths must be new; use a new suffix if retrying a failed benchmark. For language-only LoRA, append `language-only` as the fourth argument to both benchmark commands and use new output paths.

## 5. Send back the small reports first

Return `prepare.log`, `cloud_files.json`, `complete_plan/summary.json`, `abmil/selector.json`, `tiles/tiles.json`, `medgemma_inputs/student_inputs.json`, `benchmark_comparison.json`, and both benchmark directories' `run_manifest.json`, `completed.json`, `rank-*_timing.json`, `rank-*_performance.json` and `.log` files. No raw WSI transfer back is needed for this check.

After the real GPU tests pass, choose the device count for the full 60-epoch training. The existing README contains the full-run command and checkpoint schedule; it is not started automatically by this preparation script.
