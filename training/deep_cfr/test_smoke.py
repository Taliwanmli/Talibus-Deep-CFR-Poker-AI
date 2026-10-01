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

from experiment import LocalRoots, PathRef, load_config, parse_config, resolve_config
import run_smoke as smoke


class SmokeTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(smoke.DEFAULT_CONFIG)
        self.resolved = resolve_config(self.config)
        self.payload = self.config.to_dict()

    def test_optional_evaluation_preserves_existing_example(self):
        old = load_config(smoke.DEFAULT_CONFIG.with_name("example.json"))
        self.assertNotIn("evaluation", old.to_dict())
        self.assertIsNone(old.evaluation)
        self.assertEqual(parse_config(self.payload), self.config)

    def test_evaluation_validation_rejects_bad_fields(self):
        cases = [("seed", -1), ("model_seat", 6), ("hands", 0), ("workers", True),
                 ("opponent", "invalid"), ("unexpected", 1)]
        for field, value in cases:
            obj = copy.deepcopy(self.payload)
            obj["evaluation"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                parse_config(obj)

    def test_smoke_rejects_unbounded_or_non_cpu_intent(self):
        cases = [("execution", "device", "auto"), ("game", "starting_stack", 2000),
                 ("model", "hidden_dim", 512), ("training", "buffer_size", 500000),
                 ("traversal", "workers", 0), ("traversal", "consolidate_processes", False),
                 ("evaluation", "hands", 10000), ("run", "strategy_every", 0)]
        for group, field, value in cases:
            obj = copy.deepcopy(self.payload)
            obj[group][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "CPU smoke"):
                smoke.validate_smoke(resolve_config(parse_config(obj)))

    def test_training_and_evaluation_commands_propagate_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            roots = LocalRoots(smoke.REPO_ROOT, Path(directory))
            pipeline = roots.run / "pipeline"
            command = smoke.training_command(self.resolved, roots, pipeline, roots.repo / "traversal")
            def value(flag):
                return command[command.index(flag) + 1]
            self.assertEqual(value("--seed"), "42")
            self.assertEqual(value("--device"), "cpu")
            self.assertEqual(value("--num-players"), "6")
            self.assertEqual(value("--training-steps"), "2")
            self.assertEqual(value("--hidden-dim"), "32")
            self.assertEqual(value("--work-dir"), str(pipeline))
            self.assertIn("--keep-samples", command)
            self.assertIn("--traversal-consolidate-processes", command)
            self.assertNotIn("--resume", command)
            command = smoke.evaluation_command(self.resolved, roots, pipeline / "strategy.onnx", roots.repo / "eval")
            self.assertEqual(value("--seed"), "43")
            self.assertEqual(value("--hands"), "12")
            self.assertEqual(value("--deck-samples"), "2")
            self.assertEqual(value("--model-seat"), "0")

    def test_portable_commands_do_not_disclose_root_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            roots = LocalRoots(smoke.REPO_ROOT, Path(directory))
            actual = smoke.portable_command([sys.executable, str(roots.repo / "training/run.py"),
                                            str(roots.run / "output"), "/external/example"], roots)
            self.assertEqual(actual, ["python", "repo:training/run.py", "run:output", "external-executable:example"])

    def test_existing_destination_is_unchanged_and_launches_no_workload(self):
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "cpu-smoke" / "existing"
            existing.mkdir(parents=True)
            manifest = existing / "manifest.json"
            manifest.write_bytes(b"prior-run-sentinel")
            with patch.object(smoke, "record_runtime") as runtime, patch.object(smoke, "run_stage") as run:
                with contextlib.redirect_stderr(io.StringIO()):
                    code = smoke.main(["--run-root", directory, "--run-id", "existing"])
            self.assertEqual(code, 1)
            self.assertEqual(manifest.read_bytes(), b"prior-run-sentinel")
            runtime.assert_not_called()
            run.assert_not_called()

    def test_failed_stage_leaves_failed_manifest_in_owned_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(smoke, "record_runtime", return_value={}), \
                 patch.object(smoke, "run_stage", side_effect=RuntimeError("expected failure")):
                with contextlib.redirect_stderr(io.StringIO()):
                    code = smoke.main(["--run-root", directory, "--run-id", "failed"])
            destination = Path(directory) / "cpu-smoke/failed"
            manifest = json.loads((destination / "manifest.json").read_text())
            self.assertEqual(code, 1)
            self.assertEqual(manifest["phase"], "failed")
            self.assertEqual(manifest["execution"]["failure_type"], "RuntimeError")
            self.assertNotIn(directory, json.dumps(manifest))
            self.assertTrue((destination / "resolved_config.json").is_file())

    def test_real_process_failure_captures_exit_code_and_log(self):
        with tempfile.TemporaryDirectory() as directory:
            roots = LocalRoots(smoke.REPO_ROOT, Path(directory))
            output = roots.run / "failure.log"
            stages = []
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "exit code 3"):
                smoke.run_stage("failure", [sys.executable, "-c", "print('evidence');raise SystemExit(3)"],
                                roots.repo, dict(os.environ), output, stages, roots)
            self.assertEqual(stages[0]["exit_code"], 3)
            self.assertEqual(stages[0]["status"], "failed")
            self.assertIn("evidence", output.read_text())

    def test_real_process_timeout_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            roots = LocalRoots(smoke.REPO_ROOT, Path(directory))
            stages = []
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "time budget"):
                smoke.run_stage("timeout", [sys.executable, "-c", "import time;time.sleep(10)"],
                                roots.repo, dict(os.environ), roots.run / "timeout.log", stages, roots, timeout=0.1)
            self.assertEqual(stages[0]["status"], "failed")
            self.assertEqual(stages[0]["failure"], "timeout_or_interruption")

    def test_missing_executable_stage_records_launch_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            roots = LocalRoots(smoke.REPO_ROOT, Path(directory))
            stages = []
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "cannot launch"):
                smoke.run_stage("missing", [str(roots.run / "nonexistent-command")], roots.repo,
                                dict(os.environ), roots.run / "missing.log", stages, roots)
            self.assertEqual(stages[0]["status"], "failed")
            self.assertEqual(stages[0]["failure"], "process_launch")

    def test_missing_package_reports_install_command_and_failed_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(smoke, "record_runtime", side_effect=smoke.metadata.PackageNotFoundError("onnx")):
                error = io.StringIO()
                with contextlib.redirect_stderr(error):
                    code = smoke.main(["--run-root", directory, "--run-id", "missing-package"])
            self.assertEqual(code, 1)
            self.assertIn("pip install -r training/deep_cfr/requirements.txt", error.getvalue())
            manifest = json.loads((Path(directory) / "cpu-smoke/missing-package/manifest.json").read_text())
            self.assertEqual(manifest["phase"], "failed")

    def evaluation_result(self):
        return {"hands": 12, "num_players": 6, "model_seat": 0, "opponent": "random", "mode": "vs_opponent",
                "model_wins": 4, "model_losses": 4, "ties": 4, "per_seat_bb_per_hand": [0.0] * 6,
                "per_seat_bb_per_100": [0.0] * 6, "model_bb_per_hand": 0.0,
                "model_bb_per_100": 0.0, "elapsed_sec": 1.0,
                "zero_sum_check_bb_per_hand": 0.0}

    def test_rust_mode_spelling_and_result_validation(self):
        result = self.evaluation_result()
        parsed = smoke.parse_evaluation("RING_EVAL_JSON " + json.dumps(result), self.resolved)
        self.assertEqual(parsed["result"]["mode"], "vs_opponent")
        self.assertIn("pipeline-health", parsed["purpose"])
        for field, value in (("hands", 13), ("model_wins", 5), ("zero_sum_check_bb_per_hand", 0.1),
                             ("zero_sum_check_bb_per_hand", float("nan")),
                             ("per_seat_bb_per_hand", [0.0]), ("model_bb_per_100", float("inf"))):
            invalid = dict(result, **{field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                smoke.parse_evaluation("RING_EVAL_JSON " + json.dumps(invalid), self.resolved)
        for log in ("", "RING_EVAL_JSON {}\nRING_EVAL_JSON {}"):
            with self.assertRaises(ValueError):
                smoke.parse_evaluation(log, self.resolved)

    def test_calling_station_rust_result_spelling(self):
        payload = copy.deepcopy(self.payload)
        payload["evaluation"]["opponent"] = "calling-station"
        resolved = resolve_config(parse_config(payload))
        result = dict(self.evaluation_result(), opponent="calling_station")
        smoke.parse_evaluation("RING_EVAL_JSON " + json.dumps(result), resolved)

    def test_artifact_reference_hash_matches_generated_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            roots = LocalRoots(smoke.REPO_ROOT, Path(directory))
            (roots.run / "model.onnx").write_bytes(b"generated artifact")
            record = smoke.artifact_record("smoke_model", PathRef("run", "model.onnx"), roots)
            self.assertEqual(record["size_bytes"], len(b"generated artifact"))
            self.assertEqual(record["sha256"], smoke.hashlib.sha256(b"generated artifact").hexdigest())


if __name__ == "__main__":
    unittest.main()
