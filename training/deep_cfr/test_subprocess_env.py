from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from subprocess_env import build_subprocess_env, resolve_ort_dylib_path


class RuntimeLibraryTests(unittest.TestCase):
    def test_discovers_platform_library_without_selecting_provider_helpers(self):
        for platform, name in (("darwin", "libonnxruntime.1.30.0.dylib"),
                               ("linux", "libonnxruntime.so.1.30.0"),
                               ("win32", "onnxruntime.dll")):
            with self.subTest(platform=platform), tempfile.TemporaryDirectory() as directory:
                package = Path(directory)
                capi = package / "capi"
                capi.mkdir()
                library = capi / name
                library.touch()
                (capi / "libonnxruntime_providers_shared.so").touch()
                module = types.SimpleNamespace(__file__=str(package / "__init__.py"))
                with patch.object(sys, "platform", platform), patch.dict(
                    sys.modules, {"onnxruntime": module}
                ), patch.dict(os.environ, {}, clear=True):
                    self.assertEqual(resolve_ort_dylib_path(), str(library.resolve()))

    def test_explicit_override_and_existing_ort_path_take_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / "custom.dylib"
            library.touch()
            with patch.dict(os.environ, {"DEEP_CFR_ORT_DYLIB_PATH": str(library)}):
                self.assertEqual(resolve_ort_dylib_path(), str(library))
                env = build_subprocess_env({"ORT_DYLIB_PATH": "explicit-library"})
                self.assertEqual(env["ORT_DYLIB_PATH"], "explicit-library")
                self.assertEqual(env["PYTHONUTF8"], "1")

    def test_missing_or_ambiguous_library_returns_none(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            capi = package / "capi"
            capi.mkdir()
            module = types.SimpleNamespace(__file__=str(package / "__init__.py"))
            with patch.object(sys, "platform", "linux"), patch.dict(
                sys.modules, {"onnxruntime": module}
            ), patch.dict(os.environ, {}, clear=True):
                self.assertIsNone(resolve_ort_dylib_path())
                (capi / "libonnxruntime.so.1.23.0").touch()
                (capi / "libonnxruntime.so.1.30.0").touch()
                self.assertIsNone(resolve_ort_dylib_path())

    def test_multiple_symlinks_to_one_library_are_unambiguous(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            capi = package / "capi"
            capi.mkdir()
            library = capi / "libonnxruntime.so.1.30.0"
            library.touch()
            try:
                (capi / "libonnxruntime.so").symlink_to(library.name)
            except OSError:
                self.skipTest("symlinks unavailable")
            module = types.SimpleNamespace(__file__=str(package / "__init__.py"))
            with patch.object(sys, "platform", "linux"), patch.dict(
                sys.modules, {"onnxruntime": module}
            ), patch.dict(os.environ, {}, clear=True):
                self.assertEqual(resolve_ort_dylib_path(), str(library.resolve()))


if __name__ == "__main__":
    unittest.main()
