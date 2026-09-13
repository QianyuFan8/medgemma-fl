# Scope 2: fixed-sample MedGemma diagnostics

This branch adds an offline experiment to the existing CRC histopathology task:
apply one frozen Gemma Scope 2 SAE to the same inputs before and after federated
fine-tuning. The implementation is ready for a VM smoke test; real MedGemma/SAE
compatibility and medical feature interpretations have **not** been validated.

## Where it connects

```text
job.py → client.py → server aggregation → FL_global_model.pt
                                              │
CRC-VAL-HE-7K → fixed samples.json              │
                    │                         │
                    ├─ base MedGemma          │
                    └─ base + global adapter ←┘
                         │
       inference_utils.load_model_and_processor
       inference_utils.prepare_inference_inputs
                         │
          decoder layer 17 output, last prompt token
                         │
                 same frozen Scope 2 SAE
                         │
      reconstruction diagnostics + paired feature CSVs
```

- `run_scope2_analysis.py` selects samples, loads models sequentially, runs
  deterministic greedy generation, and writes reports.
- `scope2_utils.py` loads and validates the SAE, attaches a temporary observation
  hook, and calculates reconstruction/feature metrics.
- `inference_utils.py` supplies the same prompt preprocessing and model loader
  used by existing inference. Its input preparation was extracted into a shared
  helper; the generation behavior remains the same.
- `model.py` still constructs MedGemma and applies the NVFlare adapter. Training,
  LoRA aggregation, and the checkpoint format do not need changes for this audit.

Only the **first forward pass during generation** is observed. We take the last
nonpadding prompt token's residual output from decoder layer index 17 (zero-based,
the 18th decoder block). Later generated tokens are ignored by the hook. This
gives both models a matching position before their generated answers can diverge.
The position is usually part of the assistant prefix, not a particular medical
word or image patch. Its token ID and decoded text are recorded for inspection.

## Run on the VM

First transfer these changes to the VM through your `scope-2` branch. Activate the
existing working virtual environment in the project directory. No new training
run is necessary: use the saved two-client checkpoint.

```bash
python run_scope2_analysis.py \
  --dataset_dir ./CRC-VAL-HE-7K \
  --tuned_model_path ./runs/smoke-two-clients/medgemma/server/simulate_job/app_server/FL_global_model.pt \
  --max_samples 20 \
  --seed 42 \
  --output_dir ./runs/scope2-smoke
```

The script uses `torch`, `numpy`, `huggingface_hub`, and `safetensors` already
present in the model environment. It does not require SAELens or an upgrade of
the working training dependencies. Hugging Face authentication/access must still
be available. First use downloads only the selected SAE config and parameter
file (about 336 MB), not the entire Scope 2 collection. SAE inference runs in
float32 on CPU; language models run one at a time using the existing 4-bit CUDA
loader. CPU language-model mode is available but is not the tested VM setup.

Choose a **new output directory** for each experiment. To reuse exactly the same
samples, pass `--manifest ./runs/scope2-smoke/samples.json`; that manifest overrides
sample selection arguments and checks image hashes. Keep `--dataset_dir` pointed
at the image root. To pin the SAE across runs, copy the resolved `sae.revision`
from `run.json` into `--sae_revision`. Keep the base model revision/cache and
software environment fixed too; greedy decoding does not promise identical
floating-point results across hardware or package changes.

After the initial two-model smoke test, a useful **source-model control** is:

```bash
python run_scope2_analysis.py \
  --dataset_dir ./CRC-VAL-HE-7K \
  --tuned_model_path ./runs/smoke-two-clients/medgemma/server/simulate_job/app_server/FL_global_model.pt \
  --manifest ./runs/scope2-smoke/samples.json \
  --with_gemma_control \
  --output_dir ./runs/scope2-with-control
```

This additionally downloads/loads `google/gemma-3-4b-it`, for which your Hugging
Face account needs access. The control uses the same images/question with its own
processor and the same quantization policy. It is a diagnostic reference, not a
perfect reproduction of the SAE training distribution. It does not automatically
establish successful SAE transfer to MedGemma.

## Outputs and interpretation

| File | What to inspect |
| --- | --- |
| `run.json` | Completion/failure status, package versions, git state, checkpoint hash, SAE config/revision/hash and limitations. |
| `samples.json` | Fixed image list, labels, hashes, prompt, selection seed. |
| `base.json`, `tuned.json` | Predictions, actual hook path, prompt positions/input hashes, and reconstruction diagnostics for each model. |
| `base_features.npz`, `tuned_features.npz` | Complete `[samples, 16384]` float32 features and ordered sample IDs. |
| `summary.json` | Accuracy, reconstruction metrics, mean L0, and counts of prediction changes. Written on success. |
| `feature_summary.csv` | All features, sorted by mean absolute paired activation change; also includes signed mean change and activation frequency. |
| `sample_comparison.csv` | Correctness groups, raw answers, reconstruction errors and L0 for individual samples. |
| `sample_feature_deltas.csv` | Up to 20 changed features per sample, ranked by absolute change; positive delta means more activation after tuning. |

With `--with_gemma_control`, corresponding `gemma_control.json` and feature arrays
are also written. The script verifies hashes of all processed input tensors,
including image pixels, between base and tuned MedGemma, and aborts on mismatch.

Reconstruction diagnostics use the **unmodified** captured residual `x`:

- `relative_squared_error = ||x - reconstruction||² / ||x||²`. Lower is better;
  aggregate values use total error divided by total activation energy.
- `cosine_similarity` describes directional agreement for each sample.
- `fraction_variance_unexplained = total squared error / sum ||x - mean(x)||²`.
  The mean is computed across this fixed sample set; values can exceed 1.
  Last-prompt-token activations may have little between-image variance, so inspect
  this together with relative error, not as a universal pass/fail threshold.
- `l0` is the number of active SAE features. A medium-L0 SAE's training target is
  not a guaranteed value on MedGemma or on this single token position.
- Zero denominators produce JSON `null`, not a misleading perfect score.

Start by checking that reconstruction is sensible and features are neither all
zero nor broadly dense. Compare the source Gemma control, base MedGemma, and tuned
MedGemma before attributing differences to learned concepts. Then inspect
`wrong_to_right` and `right_to_wrong` examples together with their feature deltas.
The 20 random samples are a smoke test, not a class-balanced accuracy benchmark.

Feature IDs identify coordinates within this **specific SAE checkpoint**. Do not
compare IDs across layers/SAEs, label a feature as a cancer concept from its ID,
or infer that a correlated feature caused a prediction. This version does not
perform feature interventions, clinical trial matching, or cross-round tracking;
those require subsequent experiments and, for rounds, saved intermediate models.

## SAE conventions and sources

The supported checkpoint family is
`google/gemma-scope-2-4b-it/resid_post/layer_17_width_16k_l0_medium`
(also selectable: layers 9, 22, 29). The verified default config specifies
`model.layers.17.output`, JumpReLU, no affine skip connection, and 16,384 features.
Its safetensors header has encoder shape `[2560, 16384]` and decoder shape
`[16384, 2560]`.

The small inference-only loader implements:

```text
pre = x @ w_enc + b_enc
features = relu(pre) * (pre > threshold)
reconstruction = features @ w_dec + b_dec
```

It applies no extra activation normalization and does not subtract `b_dec` from
the encoder input, matching the released Gemma 3 converter conventions. It
rejects unsupported configs, parameter sets, hidden dimensions, and hook shapes.

- [Google Scope 2 4B IT model card](https://huggingface.co/google/gemma-scope-2-4b-it)
- [Selected SAE config](https://huggingface.co/google/gemma-scope-2-4b-it/blob/main/resid_post/layer_17_width_16k_l0_medium/config.json)
- [SAELens Gemma 3 loading conventions](https://github.com/jbloomAus/SAELens/blob/main/sae_lens/loading/pretrained_sae_loaders.py)
- [Google MedGemma model card](https://huggingface.co/google/medgemma-4b-it)

Google released this SAE for **Gemma 3 4B IT**, not for the medically adapted
MedGemma checkpoint. Matching architecture and tensor dimensions are necessary
but insufficient evidence of compatibility. Reports intentionally retain
`transfer_status: unvalidated` even when the computation completes.

Local checks: `python3 -m unittest discover -s tests -v`. These use synthetic SAE
weights and a tiny fake model to test equations, capture timing, sample pairing,
hash checks and report generation. An additional randomly initialized, tiny
Hugging Face Gemma 3 model checks real decoder hooks through a PEFT wrapper on
CPU and verifies that observation does not change generated tokens. No pretrained
model is downloaded for these tests; they do not replace the real VM experiment.
