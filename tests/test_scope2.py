"""CPU tests with synthetic weights/models; no HF downloads or GPU experiments."""

import contextlib
import io
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

import run_scope2_analysis as runner
from scope2_utils import (
    PromptResidualCapture, Scope2SAE, feature_comparison, input_fingerprint,
    reconstruction_metrics, resolve_decoder_layer,
)


def tiny_sae():
    return Scope2SAE({"w_enc": torch.eye(2), "w_dec": torch.eye(2),
                      "b_enc": torch.tensor([0.0, 1.0]), "b_dec": torch.tensor([3.0, 4.0]),
                      "threshold": torch.tensor([1.0, 2.0])})


class Gemma3DecoderLayer(torch.nn.Module):
    def forward(self, x):
        return (x + 1,)


class FakeModel(torch.nn.Module):
    def __init__(self, tuned=False):
        super().__init__()
        self.tuned = tuned
        self.config = types.SimpleNamespace(text_config=types.SimpleNamespace(model_type="gemma3_text", hidden_size=2))
        self.language_model = torch.nn.Module()
        self.language_model.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(17)] + [Gemma3DecoderLayer()])

    def generate(self, input_ids, attention_mask, pixel_values, **kwargs):
        self.language_model.layers[17](torch.ones(1, input_ids.shape[1], 2) * (1 + self.tuned))
        self.language_model.layers[17](torch.zeros(1, 1, 2))  # Decoding must not overwrite prefill.
        return torch.cat([input_ids, torch.tensor([[7]])], dim=1)


class FakeProcessor:
    tokenizer = types.SimpleNamespace(decode=lambda ids: "<prompt-end>")

    def batch_decode(self, *args, **kwargs):
        return ["A: adipose"]


def fake_inputs(model, processor, images):
    return {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones(1, 3, dtype=torch.int64),
            "pixel_values": torch.tensor(np.array(images[0]).copy()).float()}


class Scope2Tests(unittest.TestCase):
    def test_jumprelu_bias_and_strict_threshold(self):
        z, reconstruction = tiny_sae()(torch.tensor([[1.0, 1.0], [2.0, 2.0]]))
        torch.testing.assert_close(z, torch.tensor([[0.0, 0.0], [2.0, 3.0]]))
        torch.testing.assert_close(reconstruction, torch.tensor([[3.0, 4.0], [5.0, 7.0]]))
        self.assertEqual(list(tiny_sae().parameters()), [])

    def test_shape_and_nonfinite_fail(self):
        for x in (torch.ones(1, 3), torch.tensor([[float("nan"), 0.0]])):
            with self.assertRaises(ValueError):
                tiny_sae()(x)

    def test_prefill_position_and_cleanup(self):
        layer = Gemma3DecoderLayer()
        with PromptResidualCapture(layer, 3, 1, 2) as capture:
            layer(torch.arange(6).reshape(1, 3, 2).float())
            layer(torch.zeros(1, 1, 2))
        torch.testing.assert_close(capture.activation, torch.tensor([[3.0, 4.0]]))
        self.assertFalse(layer._forward_hooks)
        with self.assertRaises(ValueError):
            with PromptResidualCapture(layer, 3, 1, 2):
                layer(torch.zeros(1, 1, 2))
        self.assertFalse(layer._forward_hooks)

    def test_layer_through_wrapper_and_dimension_guard(self):
        model = FakeModel()
        wrapper = torch.nn.Module()
        wrapper.base_model = model
        wrapper.config = model.config
        self.assertEqual(resolve_decoder_layer(wrapper, 17, 2)[0], "base_model.language_model.layers.17")
        with self.assertRaises(ValueError):
            resolve_decoder_layer(wrapper, 17, 3)

    @patch.dict(os.environ, {"DS_ACCELERATOR": "cpu"})
    def test_real_tiny_gemma_and_peft_hooks_do_not_change_generation(self):
        # If DeepSpeed is installed locally, keep its optional import on CPU too.
        from transformers import Gemma3ForCausalLM, Gemma3TextConfig
        from peft import LoraConfig, get_peft_model

        config = Gemma3TextConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1, head_dim=8,
            max_position_embeddings=32, query_pre_attn_scalar=8, sliding_window=8,
        )
        config._attn_implementation = "eager"
        torch.manual_seed(7)
        model = Gemma3ForCausalLM(config).eval()
        inputs = {"input_ids": torch.tensor([[2, 5, 6]]), "attention_mask": torch.ones(1, 3, dtype=torch.int64)}
        with torch.inference_mode():
            expected = model.generate(**inputs, max_new_tokens=2, do_sample=False)
        for current in (model, get_peft_model(model, LoraConfig(
                r=2, lora_alpha=2, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")).eval()):
            _, module = resolve_decoder_layer(current, 1, 16)
            with PromptResidualCapture(module, 3, 2, 16) as capture, torch.inference_mode():
                observed = current.generate(**inputs, max_new_tokens=2, do_sample=False)
            torch.testing.assert_close(observed, expected)
            self.assertEqual(capture.activation.shape, (1, 16))
            self.assertTrue(torch.isfinite(capture.activation).all())
            self.assertFalse(module._forward_hooks)

    def test_fingerprint_includes_pixels_and_bfloat16(self):
        before = {"input_ids": torch.tensor([[1, 2]]), "pixel_values": torch.ones(1, 2, dtype=torch.bfloat16)}
        after = {key: value.clone() for key, value in before.items()}
        self.assertEqual(input_fingerprint(before), input_fingerprint(after))
        after["pixel_values"][0, 0] = 2
        self.assertNotEqual(input_fingerprint(before), input_fingerprint(after))

    def test_reconstruction_metrics_and_zero_denominators(self):
        x = np.array([[1., 0.], [0., 1.]])
        rows, metrics = reconstruction_metrics(x, x, x)
        self.assertEqual(metrics["fraction_variance_unexplained"], 0)
        self.assertEqual(rows[0]["cosine_similarity"], 1)
        rows, metrics = reconstruction_metrics(x * 0, x * 0, x * 0)
        self.assertIsNone(metrics["relative_squared_error"])
        self.assertIsNone(metrics["fraction_variance_unexplained"])
        self.assertIsNone(rows[0]["cosine_similarity"])

    def test_paired_absolute_delta_does_not_cancel(self):
        stats = feature_comparison(np.array([[1., 0.], [0., 1.]]), np.array([[0., 1.], [1., 0.]]))
        np.testing.assert_equal(stats["mean_delta"], [0., 0.])
        np.testing.assert_equal(stats["mean_abs_delta"], [1., 1.])

    def test_manifest_reuse_and_image_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ADI").mkdir()
            image = root / "ADI" / "one.png"
            Image.new("RGB", (2, 2)).save(image)
            manifest = runner.make_manifest(root, 1, 42)
            saved = root / "samples.json"
            runner.write_json(saved, manifest)
            self.assertEqual(manifest, runner.make_manifest(root, 99, 0, saved))
            Image.new("RGB", (2, 2), "red").save(image)
            with self.assertRaisesRegex(ValueError, "Image changed"):
                runner.make_manifest(root, 1, 42, saved)

    def test_full_cli_artifacts_with_fake_model(self):
        fake_module = types.SimpleNamespace(
            load_model_and_processor=lambda model_path, **kw: (FakeModel(model_path.endswith(".pt")), FakeProcessor()),
            prepare_inference_inputs=fake_inputs,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ADI").mkdir()
            Image.new("RGB", (2, 2)).save(root / "ADI" / "one.png")
            checkpoint = root / "global.pt"
            checkpoint.write_bytes(b"synthetic checkpoint")
            output = root / "out"
            command = ["run_scope2_analysis.py", "--tuned_model_path", str(checkpoint), "--dataset_dir", str(root),
                       "--device", "cpu", "--output_dir", str(output), "--with_gemma_control"]
            with patch("sys.argv", command), patch.dict("sys.modules", {"inference_utils": fake_module}), \
                    patch.object(runner, "load_scope2_sae", return_value=(tiny_sae(), {"test": True})), \
                    contextlib.redirect_stdout(io.StringIO()):
                runner.main()
            self.assertEqual(json.loads((output / "run.json").read_text())["status"], "complete")
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["prediction_groups"]["both_right"], 1)
            self.assertEqual(summary["changed_feature_count"], 2)
            self.assertEqual(summary["gemma_control"]["samples"], 1)
            self.assertIn("unvalidated", summary["transfer_status"])
            with np.load(output / "base_features.npz", allow_pickle=False) as arrays:
                self.assertEqual(arrays["features"].shape, (1, 2))
            self.assertTrue((output / "sample_comparison.csv").is_file())
            self.assertTrue((output / "sample_feature_deltas.csv").is_file())


if __name__ == "__main__":
    unittest.main()
