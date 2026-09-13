from __future__ import annotations

import contextlib
import io
import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

NETWORK_LABELS = (
    "advantage_p0", "advantage_p1", "strategy", "advantage_shared", "strategy_shared",
)


class ModelInitializationTests(unittest.TestCase):
    def setUp(self) -> None:
        # Discovery must not import the workload orchestrator: preparation tests
        # verify that their CLI never loads it. Restore the module cache afterward.
        already_loaded = "run_deep_cfr" in sys.modules
        self.orchestrator = importlib.import_module("run_deep_cfr")
        if not already_loaded:
            self.addCleanup(sys.modules.pop, "run_deep_cfr", None)
        self.rng_state = torch.get_rng_state()
        self.addCleanup(torch.set_rng_state, self.rng_state)
        directory = tempfile.TemporaryDirectory(prefix="talibus_model_init_")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.args = SimpleNamespace(
            seed=42, hidden_dim=32, bottleneck_dim=16, dropout_p=0.1,
            buffer_disk_dir=None, buffer_size=64, lr=0.0005, weight_decay=0.0001,
            device="cpu", onnx_opset=17,
        )

    def make_context(self, label="strategy", *, init_seed=None):
        if init_seed is None:
            init_seed = self.orchestrator.derive_initialization_seed(self.args.seed, label)
        with contextlib.redirect_stdout(io.StringIO()):
            return self.orchestrator.initialize_training_context(
                label=label,
                network_type="strategy" if label.startswith("strategy") else "advantage",
                state_path=self.root / f"{label}.pt",
                buffer_path=self.root / f"{label}.pkl",
                onnx_path=self.root / f"{label}.onnx",
                args=self.args, device=torch.device("cpu"),
                init_seed=init_seed,
            )

    def assert_state_equal(self, expected, actual) -> None:
        self.assertEqual(expected.keys(), actual.keys())
        for key, tensor in expected.items():
            self.assertTrue(torch.equal(tensor, actual[key]), key)

    def test_same_requested_seed_ignores_ambient_rng_and_repeats(self) -> None:
        for label in NETWORK_LABELS:
            with self.subTest(label=label):
                states = []
                for ambient_seed in (123, 456, 123):
                    torch.random.default_generator.manual_seed(ambient_seed)
                    states.append(self.make_context(label).model.state_dict())
                for state in states[1:]:
                    self.assert_state_equal(states[0], state)

    def test_logical_network_seeds_are_stable_and_distinct(self) -> None:
        self.assertEqual(
            [self.orchestrator.derive_initialization_seed(42, label) for label in NETWORK_LABELS],
            [4046813888, 2722868990, 1597348583, 2654435731, 2496678289],
        )
        for base_seed in (0, 42, 2**32 - 1):
            with self.subTest(base_seed=base_seed):
                seeds = [self.orchestrator.derive_initialization_seed(base_seed, label) for label in NETWORK_LABELS]
                self.assertEqual(len(set(seeds)), len(NETWORK_LABELS))
                self.assertEqual(
                    seeds,
                    [self.orchestrator.derive_initialization_seed(base_seed, label) for label in NETWORK_LABELS],
                )
                for label, seed in zip(NETWORK_LABELS, seeds):
                    self.assertNotEqual(seed, self.orchestrator.derive_initialization_seed(base_seed ^ 1, label))

    def test_initialization_seed_controls_weights(self) -> None:
        first = self.make_context(init_seed=123).model.state_dict()
        second = self.make_context(init_seed=456).model.state_dict()
        self.assertFalse(torch.equal(first["fc1.weight"], second["fc1.weight"]))

    def test_initialization_preserves_cpu_rng_without_reseeding_cuda(self) -> None:
        before = torch.get_rng_state().clone()
        with patch("torch.cuda.manual_seed_all") as cuda_seed:
            self.make_context()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        cuda_seed.assert_not_called()

    def test_checkpoint_weights_are_authoritative_and_file_is_unchanged(self) -> None:
        expected = self.make_context(init_seed=789).model.state_dict()
        for wrapped in (False, True):
            with self.subTest(wrapped=wrapped):
                path = self.root / "strategy.pt"
                torch.save({"state_dict": expected} if wrapped else expected, path)
                checkpoint_bytes = path.read_bytes()
                for ambient_seed, init_seed in ((123, 1), (456, 2)):
                    torch.random.default_generator.manual_seed(ambient_seed)
                    before = torch.get_rng_state().clone()
                    restored = self.make_context(init_seed=init_seed)
                    self.assert_state_equal(expected, restored.model.state_dict())
                    self.assertEqual(path.read_bytes(), checkpoint_bytes)
                    self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_invalid_checkpoint_still_fails_without_altering_rng(self) -> None:
        path = self.root / "strategy.pt"
        torch.save({"state_dict": {"fc1.weight": torch.zeros(1)}}, path)
        before = torch.get_rng_state().clone()
        with self.assertRaises(RuntimeError):
            self.make_context()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_advantage_subprocess_receives_explicit_initialization_seed(self) -> None:
        for label in ("advantage_p0", "advantage_p1", "advantage_shared"):
            with self.subTest(label=label):
                seed = self.orchestrator.derive_initialization_seed(self.args.seed, label)
                with patch.object(self.orchestrator, "run_streaming_command") as run:
                    self.orchestrator.train_init_model(
                        Path("train.py"), self.root,
                        self.root / f"{label}.onnx", self.root / f"{label}.pt", self.args,
                        init_seed=seed,
                    )
                command = run.call_args.args[0]
                self.assertEqual(command[command.index("--seed") + 1], str(seed))
                self.assertIn("--init-only", command)


if __name__ == "__main__":
    unittest.main()
