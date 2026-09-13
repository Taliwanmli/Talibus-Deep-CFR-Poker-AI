from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from unittest.mock import patch

from experiment import LocalRoots, PathRef, load_config, resolve_config
from provenance import artifact_record, build_manifest, canonical_json, collect_git, collect_runtime

EXAMPLE = Path(__file__).resolve().parent / "experiments" / "example.json"
SHA = "a" * 40


class ProvenanceTests(unittest.TestCase):
    def test_known_commit_and_clean_state(self):
        root = Path.cwd().resolve()
        with patch("provenance.subprocess.run", side_effect=[
            subprocess.CompletedProcess([], 0, str(root), ""),
            subprocess.CompletedProcess([], 0, SHA, ""),
            subprocess.CompletedProcess([], 0, "", ""),
        ]) as run:
            self.assertEqual(collect_git(root), {"commit": SHA, "dirty": False, "unavailable_reason": None})
        for call in run.call_args_list:
            self.assertEqual(call.args[0][:3], ["git", "-C", str(root)])
            self.assertEqual(call.kwargs["timeout"], 5)

    @unittest.skipUnless(shutil.which("git"), "Git is required for temporary checkout checks")
    def test_real_checkout_staged_unstaged_untracked_and_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()

            def git(*args):
                return subprocess.run(["git", "-C", str(root), *args], check=True,
                                      capture_output=True, text=True).stdout.strip()

            git("init", "--quiet")
            (root / "tracked").write_text("original", encoding="utf-8")
            (root / ".gitignore").write_text("ignored\n", encoding="utf-8")
            git("add", ".")
            git("-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                "-c", "commit.gpgsign=false", "commit", "-m", "fixture", "--quiet")
            expected = git("rev-parse", "HEAD")
            self.assertEqual(collect_git(root)["commit"], expected)
            self.assertFalse(collect_git(root)["dirty"])
            (root / "ignored").write_text("ignored", encoding="utf-8")
            self.assertFalse(collect_git(root)["dirty"])
            (root / "tracked").write_text("changed", encoding="utf-8")
            self.assertTrue(collect_git(root)["dirty"])
            git("add", "tracked")
            self.assertTrue(collect_git(root)["dirty"])
            git("restore", "--staged", "tracked")
            git("restore", "tracked")
            (root / "untracked").write_text("new", encoding="utf-8")
            self.assertTrue(collect_git(root)["dirty"])

    def test_not_checkout_missing_git_and_timeout(self):
        for effect, expected in (
            (FileNotFoundError(), "git_not_found"),
            (subprocess.TimeoutExpired("git", 5), "git_timeout"),
            (PermissionError(), "git_unavailable"),
        ):
            with patch("provenance.subprocess.run", side_effect=effect):
                self.assertEqual(collect_git(Path.cwd()), {
                    "commit": None, "dirty": None, "unavailable_reason": expected,
                })
        with patch("provenance.subprocess.run", return_value=subprocess.CompletedProcess([], 128, "", "private")):
            self.assertEqual(collect_git(Path.cwd())["unavailable_reason"], "not_a_checkout")

    @unittest.skipUnless(shutil.which("git"), "Git is required for a real outside-checkout check")
    def test_real_non_git_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(collect_git(Path(directory)), {
                "commit": None, "dirty": None, "unavailable_reason": "not_a_checkout",
            })

    def test_partial_evidence_preserved(self):
        root = Path.cwd().resolve()
        for head_result, status_result, expected_commit, expected_dirty in (
            (subprocess.CompletedProcess([], 0, SHA, ""), subprocess.TimeoutExpired("git", 5), SHA, None),
            (subprocess.CompletedProcess([], 128, "", ""), subprocess.CompletedProcess([], 0, "?? new", ""), None, True),
            (subprocess.CompletedProcess([], 0, "unknown", ""), subprocess.CompletedProcess([], 0, "", ""), None, False),
        ):
            with patch("provenance.subprocess.run", side_effect=[
                subprocess.CompletedProcess([], 0, str(root), ""), head_result, status_result,
            ]):
                evidence = collect_git(root)
            self.assertEqual(evidence["commit"], expected_commit)
            self.assertEqual(evidence["dirty"], expected_dirty)
            self.assertIsNotNone(evidence["unavailable_reason"])

    def test_parent_checkout_is_not_misattributed(self):
        root = Path.cwd().resolve()
        with patch("provenance.subprocess.run", return_value=subprocess.CompletedProcess([], 0, str(root.parent), "")):
            evidence = collect_git(root)
        self.assertIsNone(evidence["commit"])
        self.assertEqual(evidence["unavailable_reason"], "not_repository_root")

    def test_runtime_versions_and_missing_package(self):
        with patch("provenance.metadata.version", side_effect=["1.2.3", metadata.PackageNotFoundError("torch")]):
            runtime, warnings = collect_runtime()
        self.assertEqual(runtime["packages"], {"numpy": "1.2.3", "torch": None})
        self.assertEqual(warnings, [{"field": "runtime.packages.torch", "reason": "not_installed"}])
        self.assertEqual(set(runtime), {"python_implementation", "python_version", "system", "release", "machine", "packages"})

    def test_artifact_size_hash_external_root_and_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            roots = LocalRoots(root / "checkout", root / "external")
            roots.run.mkdir()
            (roots.run / "sample.bin").write_bytes(b"abc")
            record = artifact_record("input", PathRef("run", "sample.bin"), roots)
            self.assertEqual(record, {
                "role": "input", "root": "run", "path": "sample.bin", "size_bytes": 3,
                "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            })
            self.assertNotIn(str(root), canonical_json(record))
            with self.assertRaisesRegex(ValueError, "artifact.*readable file"):
                artifact_record("input", PathRef("run", "missing"), roots)
            with self.assertRaisesRegex(ValueError, "readable file"):
                artifact_record("input", PathRef("run", "."), roots)

    def test_artifact_symlink_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            roots = LocalRoots(root / "repo", root / "run")
            roots.run.mkdir()
            outside = root / "outside"
            outside.write_bytes(b"private")
            try:
                (roots.run / "link").symlink_to(outside)
            except OSError:
                self.skipTest("symlinks unavailable")
            with self.assertRaisesRegex(ValueError, "escapes"):
                artifact_record("input", PathRef("run", "link"), roots)
            with self.assertRaisesRegex(ValueError, "traversal"):
                PathRef("run", "../outside")

    def test_manifest_injected_observations_are_stable(self):
        resolved = resolve_config(load_config(EXAMPLE))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            roots = LocalRoots(root, root)
            (root / "a").write_bytes(b"a")
            (root / "b").write_bytes(b"b")
            git = {"commit": None, "dirty": None, "unavailable_reason": "not_a_checkout"}
            warnings = [{"field": "runtime.packages.torch", "reason": "not_installed"}]
            kwargs = dict(run_id="prepared-test", roots=roots,
                          clock=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc),
                          git_collector=lambda _: git, runtime_collector=lambda: ({"packages": {"torch": None}}, warnings))
            artifacts = [("output", PathRef("run", "b")), ("input", PathRef("run", "a"))]
            first = build_manifest(resolved, artifacts=artifacts, **kwargs)
            second = build_manifest(resolved, artifacts=reversed(artifacts), **kwargs)
            text = canonical_json(first)
            self.assertEqual(text, canonical_json(second))
            self.assertEqual(text, canonical_json(json.loads(text)))
            self.assertEqual(first["phase"], "prepared")
            self.assertEqual(first["captured_at_utc"], "2026-01-01T00:00:00+00:00")
            self.assertEqual(first["resolved_config"], resolved.to_dict())
            self.assertEqual(len(first["collection_warnings"]), 2)
            self.assertEqual(len(warnings), 1)
            self.assertNotIn(str(root), text)
            self.assertEqual(set(first), {"schema_version", "run_id", "phase", "captured_at_utc",
                                         "resolved_config", "git", "runtime", "artifacts", "collection_warnings"})
            with self.assertRaisesRegex(ValueError, "run_id"):
                build_manifest(resolved, **{**kwargs, "run_id": "../invalid"})
            with self.assertRaisesRegex(ValueError, "timezone-aware"):
                build_manifest(resolved, **{**kwargs, "clock": lambda: datetime(2026, 1, 1)})


if __name__ == "__main__":
    unittest.main()
