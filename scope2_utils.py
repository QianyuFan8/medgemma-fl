"""Offline Scope 2 diagnostics; independent of NVFlare training.

Supports only the published Gemma Scope 2 4B IT residual JumpReLU SAEs.
Parameter conventions: Google's Scope 2 release and SAELens' gemma_3 converter
(links and equations in docs/scope2.md). No activation normalization or decoder
bias subtraction is applied before encoding these released weights.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch

SAE_REPO = "google/gemma-scope-2-4b-it"
SAE_SOURCE_MODEL = "google/gemma-3-4b-it"


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_fingerprint(inputs) -> str:
    """Include token IDs, masks AND processed pixels to check paired inputs."""
    digest = hashlib.sha256()
    for name, value in sorted(inputs.items()):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Unexpected non-tensor model input: {name}")
        tensor = value.detach().cpu().contiguous()
        digest.update(f"{name}:{tensor.dtype}:{list(tensor.shape)}".encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class Scope2SAE(torch.nn.Module):
    """Frozen, float32 inference for the release's five-tensor JumpReLU SAE."""

    def __init__(self, tensors: dict[str, torch.Tensor]):
        super().__init__()
        expected = {"w_enc", "w_dec", "b_enc", "b_dec", "threshold"}
        if set(tensors) != expected:
            raise ValueError(f"Unsupported SAE tensor keys: {set(tensors)}; expected {expected}")
        enc = tensors["w_enc"]
        if enc.ndim != 2:
            raise ValueError("SAE w_enc must be a matrix")
        self.d_in, self.width = enc.shape
        shapes = {"w_enc": (self.d_in, self.width), "w_dec": (self.width, self.d_in),
                  "b_enc": (self.width,), "b_dec": (self.d_in,), "threshold": (self.width,)}
        for name, shape in shapes.items():
            value = tensors[name].detach().float()
            if tuple(value.shape) != shape or not torch.isfinite(value).all():
                raise ValueError(f"Invalid shape or nonfinite values in SAE {name}")
            self.register_buffer(name, value)
        if (self.threshold < 0).any():
            raise ValueError("JumpReLU thresholds must be nonnegative")
        self.eval()

    @torch.inference_mode()
    def forward(self, x):
        x = x.to(device=self.w_enc.device, dtype=torch.float32)
        if x.ndim != 2 or x.shape[-1] != self.d_in or not torch.isfinite(x).all():
            raise ValueError(f"Expected finite activations [N, {self.d_in}]")
        pre = x @ self.w_enc + self.b_enc
        features = torch.relu(pre) * (pre > self.threshold)
        reconstructed = features @ self.w_dec + self.b_dec
        if not torch.isfinite(reconstructed).all() or not torch.isfinite(features).all():
            raise ValueError("Nonfinite SAE output")
        return features, reconstructed


def load_scope2_sae(layer=17, revision="main"):
    """Resolve once to an immutable HF commit, then fetch config and weights."""
    from huggingface_hub import HfApi, hf_hub_download
    from safetensors.torch import load_file

    if layer not in (9, 17, 22, 29):
        raise ValueError("This MVP supports residual layers 9, 17, 22, 29 only")
    folder = f"resid_post/layer_{layer}_width_16k_l0_medium"
    commit = HfApi().model_info(SAE_REPO, revision=revision).sha
    config_path = hf_hub_download(SAE_REPO, f"{folder}/config.json", revision=commit)
    config = json.loads(Path(config_path).read_text())
    hook = f"model.layers.{layer}.output"
    required = {"model_name": SAE_SOURCE_MODEL, "type": "sae", "architecture": "jump_relu",
                "affine_connection": False, "width": 16384,
                "hf_hook_point_in": hook, "hf_hook_point_out": hook}
    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError(f"Unsupported SAE config {key}: {config.get(key)!r}; expected {value!r}")
    weights_path = hf_hub_download(SAE_REPO, f"{folder}/params.safetensors", revision=commit)
    sae = Scope2SAE(load_file(weights_path, device="cpu"))
    if sae.width != config["width"] or sae.d_in != 2560:
        raise ValueError("Unexpected Gemma 3 4B SAE dimensions")
    metadata = {"repo": SAE_REPO, "revision": commit, "folder": folder, "config": config,
                "weights_sha256": file_sha256(weights_path), "d_in": sae.d_in,
                "normalization": "none", "subtract_b_dec_before_encoding": False,
                "compute_dtype": "float32", "device": "cpu"}
    return sae, metadata


def resolve_decoder_layer(model, layer: int, d_in: int):
    """Find the language decoder through either the base or PEFT wrapper."""
    config = model.config
    text_config = getattr(config, "text_config", config)
    if getattr(text_config, "model_type", None) != "gemma3_text":
        raise ValueError("Expected a Gemma 3 text decoder; other architectures are unsupported")
    if getattr(text_config, "hidden_size", None) != d_in:
        raise ValueError("Model hidden size does not match SAE input size")
    candidates = [(name, module) for name, module in model.named_modules()
                  if re.search(rf"(?:^|\.)layers\.{layer}$", name)
                  and module.__class__.__name__ == "Gemma3DecoderLayer"]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one Gemma3DecoderLayer at index {layer}; found "
                         f"{[name for name, _ in candidates]}")
    return candidates[0]


class PromptResidualCapture:
    """Observe the first generation forward (prefill); never alter its output."""

    def __init__(self, module, sequence_length: int, token_position: int, d_in: int):
        self.module = module
        self.sequence_length = sequence_length
        self.token_position = token_position
        self.d_in = d_in
        self.activation = None

    def _hook(self, module, args, output):
        if self.activation is not None:
            return  # Subsequent autoregressive tokens must not replace prompt activations.
        hidden = output[0] if isinstance(output, tuple) else output
        expected = (1, self.sequence_length, self.d_in)
        if not isinstance(hidden, torch.Tensor) or tuple(hidden.shape) != expected:
            raise ValueError(f"Unexpected prefill residual shape; expected {expected}")
        if not 0 <= self.token_position < self.sequence_length:
            raise ValueError("Prompt position is outside sequence")
        self.activation = hidden[:, self.token_position, :].detach().float().cpu().clone()

    def __enter__(self):
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.handle.remove()
        if exc_type is None and self.activation is None:
            raise RuntimeError("Target decoder hook was never called")


def reconstruction_metrics(x: np.ndarray, reconstruction: np.ndarray, features: np.ndarray):
    """Return per-sample diagnostics plus aggregate metrics; no pass/fail claim."""
    x, reconstruction = np.asarray(x, dtype=np.float64), np.asarray(reconstruction, dtype=np.float64)
    if x.ndim != 2 or x.shape != reconstruction.shape or features.shape[0] != len(x) or not len(x):
        raise ValueError("Invalid paired reconstruction arrays")
    error = ((x - reconstruction) ** 2).sum(axis=1)
    energy = (x ** 2).sum(axis=1)
    norm_product = np.linalg.norm(x, axis=1) * np.linalg.norm(reconstruction, axis=1)
    rows = []
    for i in range(len(x)):
        rows.append({"relative_squared_error": float(error[i] / energy[i]) if energy[i] > 0 else None,
                     "cosine_similarity": float(x[i] @ reconstruction[i] / norm_product[i])
                     if norm_product[i] > 0 else None,
                     "l0": int(np.count_nonzero(features[i]))})
    centered_energy = ((x - x.mean(axis=0)) ** 2).sum()
    summary = {"relative_squared_error": float(error.sum() / energy.sum()) if energy.sum() > 0 else None,
               "fraction_variance_unexplained": float(error.sum() / centered_energy)
               if centered_energy > 0 else None,
               "mean_l0": float(np.count_nonzero(features, axis=1).mean()),
               "zero_feature_samples": int((np.count_nonzero(features, axis=1) == 0).sum())}
    return rows, summary


def feature_comparison(base: np.ndarray, tuned: np.ndarray):
    if base.shape != tuned.shape or base.ndim != 2 or not len(base):
        raise ValueError("Expected nonempty, paired [samples, features] arrays")
    delta = tuned - base
    return {"base_mean": base.mean(axis=0), "tuned_mean": tuned.mean(axis=0),
            "mean_delta": delta.mean(axis=0), "mean_abs_delta": np.abs(delta).mean(axis=0),
            "base_active_fraction": (base > 0).mean(axis=0),
            "tuned_active_fraction": (tuned > 0).mean(axis=0)}
