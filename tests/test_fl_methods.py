import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from fl_methods import (SCAFFOLD_WIRE_PREFIX, FLMethodState, answer_offsets, fedlc_adjust, fedlc_margins,
                        is_scaffold_key, prox_penalty, scaffold_linear_term, scaffold_update_ci, shifted_answer_ce,
                        snapshot, split_scaffold_params, trainable_named_params)


@unittest.skipUnless(torch is not None, "torch not installed")
class FLMethodTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 1))
        self.model[1].weight.requires_grad_(False)
        self.params = trainable_named_params(self.model)

    def test_split_and_keys(self):
        params = {"model.a.lora_A.weight": torch.ones(1), SCAFFOLD_WIRE_PREFIX + "a.lora_A.default.weight": torch.zeros(1)}
        adapter, control = split_scaffold_params(params)
        self.assertEqual(list(adapter), ["model.a.lora_A.weight"])
        self.assertEqual(list(control), ["a.lora_A.default.weight"])
        self.assertTrue(is_scaffold_key(SCAFFOLD_WIRE_PREFIX + "x"))
        self.assertFalse(is_scaffold_key("model.base_model.x"))

    def test_prox_penalty_and_gradient(self):
        anchor = snapshot(self.params)
        self.assertEqual(float(prox_penalty(self.params, anchor, 0.05)), 0.0)
        with torch.no_grad():
            self.params["0.weight"].add_(1.0)
        pen = prox_penalty(self.params, anchor, 0.05)
        self.assertAlmostEqual(float(pen), 0.5 * 0.05 * 6.0, places=5)  # 6 elements moved by 1
        pen.backward()
        self.assertTrue(torch.allclose(self.params["0.weight"].grad, torch.full((2, 3), 0.05)))
        self.assertTrue(set(self.params) == {"0.weight", "0.bias", "1.bias"})

    def test_scaffold_linear_term_gradient(self):
        correction = {name: torch.full_like(p, 0.25) for name, p in self.params.items()}
        term = scaffold_linear_term(self.params, correction)
        term.backward()
        for name, p in self.params.items():
            self.assertTrue(torch.allclose(p.grad, torch.full_like(p, 0.25)), name)

    def test_scaffold_update(self):
        x = {"w": torch.tensor([1.0, 2.0])}
        y = {"w": torch.tensor([0.5, 2.5])}
        ci = scaffold_update_ci(None, None, x, y, lr_sum=0.5)
        self.assertTrue(torch.allclose(ci["w"], torch.tensor([1.0, -1.0])))
        c = {"w": torch.tensor([0.2, 0.2])}
        ci2 = scaffold_update_ci(ci, c, x, y, lr_sum=0.5)
        self.assertTrue(torch.allclose(ci2["w"], torch.tensor([1.0, -1.0]) - 0.2 + torch.tensor([1.0, -1.0])))
        with self.assertRaises(ValueError):
            scaffold_update_ci(None, None, x, y, lr_sum=0.0)

    def test_answer_offsets_and_ce(self):
        labels = torch.tensor([[-100, -100, 7, 3, -100], [-100, -100, -100, 5, 9]])
        first, offset, kept = answer_offsets(labels)
        self.assertEqual(first.tolist(), [2, 3])
        self.assertEqual((offset, kept), (1, 4))
        vocab = 10
        full = torch.randn(2, 5, vocab)
        ref = torch.nn.functional.cross_entropy(full[:, :-1].reshape(-1, vocab), labels[:, 1:].reshape(-1), ignore_index=-100)
        ours = shifted_answer_ce(full[:, offset:], labels, offset)
        self.assertAlmostEqual(float(ref), float(ours), places=5)

    def test_fedlc_margins_and_adjust(self):
        m = fedlc_margins({"ERMS": 16, "ARMS": 1, "X": 0}, tau=1.0)
        self.assertAlmostEqual(m["ERMS"], 0.5)
        self.assertAlmostEqual(m["ARMS"], 1.0)
        self.assertGreater(m["X"], m["ARMS"])  # absent class gets the largest handicap
        row = torch.zeros(5)
        fedlc_adjust(row, {"ERMS": 1, "ARMS": 3}, {"ERMS": 0.5, "ARMS": 1.0})
        self.assertEqual(row.tolist(), [0.0, -0.5, 0.0, -1.0, 0.0])

    def test_state_round_trip(self):
        state = FLMethodState("scaffold")
        state.start_round(self.model, server_control=None)
        self.assertIsNone(state.correction)  # first round: no server c, no c_i
        state.lr_sum = 0.1
        with torch.no_grad():
            self.params["0.bias"].add_(0.1)
        wire = state.end_round(self.model)
        self.assertEqual(set(wire), {SCAFFOLD_WIRE_PREFIX + k for k in self.params})
        self.assertTrue(torch.allclose(wire[SCAFFOLD_WIRE_PREFIX + "0.bias"], torch.full((2,), -1.0)))
        # second round with the server average -> a correction exists and the term is differentiable
        server_c = {k[len(SCAFFOLD_WIRE_PREFIX):]: v for k, v in wire.items()}
        state.start_round(self.model, server_control=server_c)
        self.assertIsNotNone(state.correction)
        self.assertTrue(torch.allclose(state.correction["0.bias"], torch.zeros(2)))  # c == c_i after one client
        with self.assertRaises(ValueError):
            FLMethodState("nope")
        prox = FLMethodState("fedprox", mu=0.1)
        prox.start_round(self.model, None)
        self.assertEqual(float(prox.extra_term()), 0.0)
        self.assertEqual(prox.end_round(self.model), {})
        with self.assertRaises(ValueError):
            FLMethodState("fedlc", classes=["ERMS", "ARMS"]).start_round(self.model, None)


if __name__ == "__main__":
    unittest.main()
