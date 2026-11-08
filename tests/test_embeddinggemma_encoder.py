"""EmbeddingGemma 2 tiles go through SentenceTransformer.encode, mocked here."""
import sys
import types
import unittest

import numpy as np
from PIL import Image

from data_preprocessing.wsi_embed import EmbeddingGemmaImageEncoder, load_encoder

torch = __import__("torch")


class FakeSentenceTransformer:
    """Accepts only the input form named by `accepts`, like a given library release."""

    accepts = "pil"
    dim = 768
    fill = 1.0
    calls = []

    def __init__(self, name, device=None, model_kwargs=None, config_kwargs=None):
        FakeSentenceTransformer.calls.append(dict(name=name, device=device, model_kwargs=model_kwargs,
                                                  config_kwargs=config_kwargs))

    def eval(self):
        return self

    def encode(self, inputs, batch_size, convert_to_numpy, normalize_embeddings, show_progress_bar):
        assert normalize_embeddings and convert_to_numpy
        first = inputs[0]
        if self.accepts == "pil" and not isinstance(first, Image.Image):
            raise TypeError("expects PIL images")
        if self.accepts == "dict" and not isinstance(first, dict):
            raise TypeError("expects dict inputs")
        values = np.full((len(inputs), self.dim), self.fill, dtype=np.float32)
        return values / np.linalg.norm(values, axis=1, keepdims=True)


def fake_module():
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    module.__version__ = "test"
    return module


class EmbeddingGemmaEncoderTests(unittest.TestCase):
    def setUp(self):
        FakeSentenceTransformer.accepts, FakeSentenceTransformer.dim, FakeSentenceTransformer.fill = "pil", 768, 1.0
        FakeSentenceTransformer.calls = []
        # Swap only this one module. patch.dict(sys.modules) would also evict every
        # torch/timm submodule imported during the test, and re-importing those
        # later breaks other test files (duplicate TORCH_LIBRARY registration).
        previous = sys.modules.get("sentence_transformers")
        sys.modules["sentence_transformers"] = fake_module()
        self.addCleanup(lambda: sys.modules.__setitem__("sentence_transformers", previous) if previous
                        else sys.modules.pop("sentence_transformers", None))
        self.device = torch.device("cpu")

    def test_encodes_pil_tiles_with_fp32_on_cpu_and_no_audio(self):
        encoder = EmbeddingGemmaImageEncoder("google/embeddinggemma-2", self.device)
        call = FakeSentenceTransformer.calls[0]
        self.assertEqual(call["model_kwargs"], {"torch_dtype": torch.float32})
        self.assertEqual(call["config_kwargs"], {"audio_config": None})
        values = encoder([Image.new("RGB", (224, 224))] * 3)
        self.assertEqual(values.shape, (3, 768))
        np.testing.assert_allclose(np.linalg.norm(values, axis=1), 1.0, rtol=1e-5)
        self.assertEqual(encoder.provenance["embedding_api"], "sentence_transformers.encode")
        self.assertEqual(encoder.provenance["image_input_style"], "pil")

    def test_falls_back_to_dict_inputs_and_keeps_that_form(self):
        FakeSentenceTransformer.accepts = "dict"
        encoder = EmbeddingGemmaImageEncoder("google/embeddinggemma-2", self.device)
        self.assertEqual(encoder.input_style, "dict")
        self.assertEqual(encoder([Image.new("RGB", (224, 224))]).shape, (1, 768))

    def test_wrong_dimension_is_rejected(self):
        FakeSentenceTransformer.dim = 512
        with self.assertRaises(RuntimeError):
            EmbeddingGemmaImageEncoder("google/embeddinggemma-2", self.device)

    def test_non_finite_embeddings_are_rejected(self):
        FakeSentenceTransformer.fill = float("nan")
        with self.assertRaises(RuntimeError):
            EmbeddingGemmaImageEncoder("google/embeddinggemma-2", self.device)

    def test_load_encoder_routes_embeddinggemma(self):
        model, encoder, dim, kind, dtype = load_encoder("google/embeddinggemma-2", self.device)
        self.assertEqual((dim, kind, dtype), (768, "embeddinggemma", torch.float32))
        self.assertIs(model, encoder.model)


if __name__ == "__main__":
    unittest.main()
