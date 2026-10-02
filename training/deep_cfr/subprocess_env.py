#!/usr/bin/env python3
"""Shared subprocess environment helpers for Deep CFR scripts."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Mapping


def resolve_ort_dylib_path() -> str | None:
    """Find the native ORT library bundled with the active Python environment."""
    override = os.environ.get("DEEP_CFR_ORT_DYLIB_PATH", "").strip()
    if override:
        override_path = Path(override)
        if override_path.is_file():
            return str(override_path)
    try:
        import onnxruntime  # type: ignore
    except Exception:
        return None

    package_root = Path(onnxruntime.__file__).resolve().parent
    capi_dir = package_root / "capi"
    if sys.platform.startswith("win"):
        candidates = [capi_dir / "onnxruntime.dll"]
    elif sys.platform == "darwin":
        candidates = sorted(capi_dir.glob("libonnxruntime*.dylib"))
    else:
        candidates = sorted(capi_dir.glob("libonnxruntime.so*"))
    # Symlinks may expose the same library under several versioned names.
    libraries = {path.resolve() for path in candidates if path.is_file()}
    if len(libraries) == 1:
        return str(libraries.pop())
    return None


def resolve_runtime_dependency_dirs() -> list[str]:
    """Resolve directories that help ORT load CUDA/cuDNN dependencies on Windows."""
    if not sys.platform.startswith("win"):
        return []
    out: list[str] = []
    try:
        import onnxruntime  # type: ignore

        capi_dir = Path(onnxruntime.__file__).resolve().parent / "capi"
        if capi_dir.exists():
            out.append(str(capi_dir))
    except Exception:
        pass
    try:
        import torch  # type: ignore

        torch_lib_dir = Path(torch.__file__).resolve().parent / "lib"
        if torch_lib_dir.exists():
            out.append(str(torch_lib_dir))
    except Exception:
        pass
    seen: set[str] = set()
    deduped: list[str] = []
    for item in out:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def build_subprocess_env(base_env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Build a subprocess environment with UTF-8 and native ORT settings."""
    env = dict(base_env) if base_env is not None else dict(os.environ)
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    ort_dylib_path = resolve_ort_dylib_path()
    if ort_dylib_path and not env.get("ORT_DYLIB_PATH"):
        env["ORT_DYLIB_PATH"] = ort_dylib_path
    dependency_dirs = resolve_runtime_dependency_dirs()
    if dependency_dirs:
        existing_path = env.get("PATH", "")
        combined = dependency_dirs + ([existing_path] if existing_path else [])
        env["PATH"] = os.pathsep.join(part for part in combined if part)
    return env
