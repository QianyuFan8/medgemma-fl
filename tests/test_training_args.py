"""CPU-only regression tests for argument construction, not GPU/TRL integration."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


def load_builder():
    # Exercise the actual function without importing CUDA/NVFlare dependencies.
    path = Path(__file__).resolve().parents[1] / "client.py"
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "_build_training_args")

    def config_stub(**kwargs):
        if "warmup_ratio" in kwargs:
            raise TypeError("Unexpected keyword argument 'warmup_ratio'")
        return kwargs

    namespace = {"SFTConfig": config_stub}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_build_training_args"]


class TrainingArgumentsTests(unittest.TestCase):
    def test_warmup_and_task_settings(self):
        builder = load_builder()
        for task in ("methylation", "histology"):
            for do_eval in (False, True):
                for max_steps in (None, 1):
                    with self.subTest(task=task, do_eval=do_eval, max_steps=max_steps):
                        args = SimpleNamespace(
                            task=task, num_train_epochs=1,
                            per_device_train_batch_size=1, per_device_eval_batch_size=1,
                            gradient_accumulation_steps=4, logging_steps=25,
                            learning_rate=2e-4, report_to="none", max_steps=max_steps,
                            max_seq_length=4096, eval_steps=50,
                        )
                        config = builder(args, "unused-test-output", do_eval)
                        self.assertNotIn("warmup_ratio", config)
                        self.assertEqual(config["warmup_steps"], 0.03)
                        self.assertEqual(config["eval_strategy"], "steps" if do_eval else "no")
                        self.assertEqual("max_steps" in config, max_steps is not None)
                        if task == "methylation":
                            self.assertEqual(config["max_length"], 4096)
                        else:
                            self.assertNotIn("max_length", config)


if __name__ == "__main__":
    unittest.main()
