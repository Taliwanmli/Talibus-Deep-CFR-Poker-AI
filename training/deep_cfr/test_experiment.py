from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiment import LocalRoots, PathRef, canonical_json, load_config, main, parse_config, resolve_config
from model import INPUT_DIM, MAX_ACTIONS, ModelConfig

EXAMPLE = Path(__file__).resolve().parent / "experiments" / "example.json"
SCRIPT = EXAMPLE.parents[1] / "experiment.py"


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.payload = json.loads(EXAMPLE.read_text(encoding="utf-8"))

    def test_example_loads(self):
        config = load_config(EXAMPLE)
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.training.training_steps, 2)
        self.assertEqual(config.to_dict(), self.payload)

    def test_all_required_fields(self):
        for field in self.payload:
            with self.subTest(field=field):
                obj = copy.deepcopy(self.payload)
                del obj[field]
                with self.assertRaisesRegex(ValueError, field):
                    parse_config(obj)
        for group, fields in self.payload.items():
            if not isinstance(fields, dict) or group == "model":
                continue
            for field in fields:
                with self.subTest(group=group, field=field):
                    obj = copy.deepcopy(self.payload)
                    del obj[group][field]
                    with self.assertRaisesRegex(ValueError, field):
                        parse_config(obj)
        for reference in ("cluster_dir", "output_dir"):
            for field in ("root", "path"):
                obj = copy.deepcopy(self.payload)
                del obj["paths"][reference][field]
                with self.assertRaisesRegex(ValueError, field):
                    parse_config(obj)

    def test_unknown_fields_and_unsupported_dimensions(self):
        for group in (None, "game", "run", "model", "training", "traversal", "execution", "export", "paths"):
            obj = copy.deepcopy(self.payload)
            target = obj if group is None else obj[group]
            target["unexpected"] = 1
            with self.subTest(group=group), self.assertRaisesRegex(ValueError, "unknown.*unexpected"):
                parse_config(obj)
        for field in ("input_dim", "max_actions"):
            obj = copy.deepcopy(self.payload)
            obj["model"][field] = 1
            with self.assertRaisesRegex(ValueError, field):
                parse_config(obj)

    def test_malformed_duplicate_and_unreadable_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for text, message in (("{", "malformed JSON"), ('{"seed":1,"seed":2}', "duplicate.*seed"),
                                  ('{"model":{"hidden_dim":1,"hidden_dim":2}}', "duplicate.*hidden_dim")):
                path.write_text(text, encoding="utf-8")
                with self.subTest(text=text), self.assertRaisesRegex(ValueError, message):
                    load_config(path)
            path.write_bytes(b"\xff")
            with self.assertRaisesRegex(ValueError, "UTF-8"):
                load_config(path)
            with self.assertRaisesRegex(ValueError, "readable"):
                load_config(Path(directory) / "missing")

    def test_wrong_root_and_group_types(self):
        for value in (None, [], True, "config", 1):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "config.*object"):
                parse_config(value)
        for group in ("game", "run", "model", "training", "traversal", "execution", "export", "paths"):
            obj = copy.deepcopy(self.payload)
            obj[group] = []
            with self.subTest(group=group), self.assertRaisesRegex(ValueError, group):
                parse_config(obj)

    def test_invalid_scalars(self):
        cases = [
            ("schema_version", True), ("schema_version", 2), ("schema_version", 1.0),
            ("name", "../bad"), ("name", ""), ("seed", True), ("seed", -1),
            ("seed", 2**32), ("seed", 1.0), ("seed", "42"),
            ("game.num_players", 7), ("game.small_blind", 0),
            ("run.iterations", False), ("run.strategy_every", -1),
            ("model.hidden_dim", 0), ("model.hidden_dim", None), ("model.dropout_p", True),
            ("model.dropout_p", 1), ("model.dropout_p", float("nan")),
            ("training.batch_size", True), ("training.training_steps", 0),
            ("training.lr", float("inf")), ("training.weight_decay", -1),
            ("training.lr", 10**400), ("training.adv_huber_delta", 0),
            ("traversal.workers", -1), ("traversal.consolidate_processes", 1),
            ("execution.device", "mps"), ("export.onnx_opset", False),
        ]
        for field, value in cases:
            obj = copy.deepcopy(self.payload)
            keys = field.split(".")
            target = obj if len(keys) == 1 else obj[keys[0]]
            target[keys[-1]] = value
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, field):
                parse_config(obj)

    def test_nonfinite_json_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for value in (float("nan"), float("inf"), -float("inf")):
                obj = copy.deepcopy(self.payload)
                obj["training"]["lr"] = value
                path.write_text(json.dumps(obj), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "training.lr"):
                    load_config(path)

    def test_incompatible_settings(self):
        for group, field, value, message in (
            ("game", "small_blind", 30, "game requires"),
            ("traversal", "seat_chunks", 9, "seat_chunks"),
            ("traversal", "consolidate_processes", True, "consolidate_processes"),
        ):
            obj = copy.deepcopy(self.payload)
            obj[group][field] = value
            with self.assertRaisesRegex(ValueError, message):
                parse_config(obj)

    def test_model_defaults_and_constants(self):
        self.payload["model"] = {}
        config = parse_config(self.payload)
        resolved = resolve_config(config)
        defaults = ModelConfig()
        for field in ("hidden_dim", "bottleneck_dim", "dropout_p"):
            self.assertEqual(getattr(resolved, field), getattr(defaults, field))
        self.assertEqual(resolved.input_dim, INPUT_DIM)
        self.assertEqual(resolved.max_actions, MAX_ACTIONS)
        self.assertEqual(config.to_dict()["model"], {})

    def test_overrides_preserve_input_and_absent_values(self):
        config = parse_config(self.payload)
        before = config.to_dict()
        self.assertEqual(resolve_config(config).intent.to_dict(), before)
        resolved = resolve_config(config, seed=0, output_dir=PathRef("run", "other"))
        self.assertEqual(resolved.intent.seed, 0)
        self.assertEqual(resolved.intent.output_dir.path, "other")
        self.assertEqual(resolved.intent.training, config.training)
        self.assertEqual(config.to_dict(), before)
        for seed in (-1, True, 2**32):
            with self.assertRaisesRegex(ValueError, "seed"):
                resolve_config(config, seed=seed)
        with self.assertRaisesRegex(ValueError, "output_dir.root"):
            resolve_config(config, output_dir=PathRef("repo", "other"))

    def test_requested_workers_remain_requests(self):
        for workers in (0, 1, 999):
            self.payload["traversal"]["workers"] = workers
            resolved = resolve_config(parse_config(self.payload)).to_dict()
            self.assertEqual(resolved["traversal"]["requested_workers"], workers)
            self.assertNotIn("actual_workers", resolved["traversal"])
            self.assertNotIn("selected_workers", resolved["traversal"])

    def test_requested_and_selected_device(self):
        for available in (True, False):
            for requested in ("auto", "cuda", "cpu"):
                self.payload["execution"]["device"] = requested
                with patch("torch.cuda.is_available", return_value=available):
                    resolved = resolve_config(parse_config(self.payload)).to_dict()
                expected = "cuda" if available and requested != "cpu" else "cpu"
                self.assertEqual(resolved["execution"], {
                    "requested_device": requested, "selected_device": expected,
                })

    def test_portable_paths_reject_escape_and_platform_specific_names(self):
        for path in ("/absolute", "C:/run", "D:\\run", "C:relative", "//server/share", "\\\\server\\share",
                     "../out", "a/../../out", "~/out", "$HOME/out", "a\\b", "NUL.txt", "a.", "a ", "a\x00b"):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "path"):
                PathRef("run", path)
        self.assertEqual(PathRef("run", "./a//b").path, "a/b")
        self.assertEqual(PathRef("run", ".").path, ".")
        with self.assertRaisesRegex(ValueError, "root"):
            PathRef("unknown", "a")

    def test_external_run_root_and_cwd_independence(self):
        config = load_config(EXAMPLE)
        before = resolve_config(config).to_dict()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            roots = LocalRoots(base / "repo", base / "external")
            expected = roots.run / "example"
            original = Path.cwd()
            try:
                os.chdir(base)
                self.assertEqual(resolve_config(config).to_dict(), before)
                self.assertEqual(roots.bind(config.output_dir), expected)
            finally:
                os.chdir(original)
            self.assertFalse(roots.repo.exists())
            self.assertFalse(roots.run.exists())
        with self.assertRaisesRegex(ValueError, "absolute"):
            LocalRoots(Path("relative"), Path("relative"))

    def test_canonical_round_trip(self):
        obj = resolve_config(load_config(EXAMPLE)).to_dict()
        text = canonical_json(obj)
        self.assertEqual(text, canonical_json(dict(reversed(list(obj.items())))))
        self.assertEqual(text, canonical_json(json.loads(text)))
        self.assertTrue(text.endswith("\n"))
        self.assertFalse(text.endswith("\n\n"))
        self.assertIn("\\u00e9", canonical_json({"label": "é"}))
        with self.assertRaises(ValueError):
            canonical_json({"x": float("nan")})

    def test_preparation_cli_launches_only_git_and_no_workload(self):
        import train
        calls = []
        popen = subprocess.Popen

        def checked_popen(command, *args, **kwargs):
            self.assertEqual(command[0], "git", f"unexpected workload launch: {command}")
            calls.append(command)
            return popen(command, *args, **kwargs)

        def forbidden(*args, **kwargs):
            self.fail("preparation attempted model work")

        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            with patch("subprocess.Popen", side_effect=checked_popen), \
                    patch("model.DeepCfrNet.__init__", side_effect=forbidden), \
                    patch.object(train, "export_onnx", side_effect=forbidden), \
                    patch("torch.save", side_effect=forbidden), contextlib.redirect_stdout(output):
                result = main(["--config", str(EXAMPLE), "--run-id", "test-preparation",
                               "--run-root", directory, "--seed", "123"])
            self.assertEqual(result, 0)
            obj = json.loads(output.getvalue())
            self.assertEqual(obj["phase"], "prepared")
            self.assertEqual(obj["resolved_config"]["seed"], 123)
            self.assertEqual(obj["artifacts"], [])
            self.assertTrue(calls)
            self.assertEqual(list(Path(directory).iterdir()), [])
            self.assertNotIn(directory, output.getvalue())
            self.assertNotIn("run_deep_cfr", sys.modules)

    def test_real_cli_and_error_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            command = [sys.executable, str(SCRIPT), "--config", str(EXAMPLE), "--run-id", "cli-test",
                       "--run-root", directory]
            result = subprocess.run(command, cwd=directory, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            obj = json.loads(result.stdout)
            self.assertEqual(result.stdout, canonical_json(obj).encode("utf-8"))
            self.assertEqual(obj["resolved_config"]["execution"]["selected_device"], "cpu")
            self.assertEqual(list(Path(directory).iterdir()), [])
            result = subprocess.run(command + ["--seed", "-1"], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 2)
            self.assertIn("seed", result.stderr)
            self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
