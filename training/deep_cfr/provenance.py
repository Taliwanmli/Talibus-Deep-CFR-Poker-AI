"""Preparation-time evidence; this module never executes an experiment."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable

if TYPE_CHECKING:
    from experiment import LocalRoots, PathRef, ResolvedConfig


def canonical_json(value: Any) -> str:
    """Stable JSON for normalized objects, including a single final newline."""
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n"


class _GitUnavailable(Exception):
    pass


def _git(repo_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=5, check=False,
        )
    except FileNotFoundError as exc:
        raise _GitUnavailable("git_not_found") from exc
    except subprocess.TimeoutExpired as exc:
        raise _GitUnavailable("git_timeout") from exc
    except OSError as exc:
        raise _GitUnavailable("git_unavailable") from exc
    if result.returncode:
        raise _GitUnavailable("git_command_failed")
    return result.stdout.strip()


def collect_git(repo_root: Path) -> dict[str, Any]:
    """Collect a full SHA and dirty state without disclosing paths or stderr."""
    evidence: dict[str, Any] = {"commit": None, "dirty": None, "unavailable_reason": None}
    repo_root = repo_root.resolve()
    try:
        checkout_root = _git(repo_root, "rev-parse", "--show-toplevel")
        if Path(checkout_root).resolve() != repo_root:
            evidence["unavailable_reason"] = "not_repository_root"
            return evidence
    except _GitUnavailable as exc:
        reason = str(exc)
        evidence["unavailable_reason"] = (
            "not_a_checkout" if reason == "git_command_failed" else reason
        )
        return evidence
    reasons = []
    try:
        commit = _git(repo_root, "rev-parse", "--verify", "HEAD")
        if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit):
            raise _GitUnavailable("invalid_commit")
        evidence["commit"] = commit
    except _GitUnavailable as exc:
        reasons.append("head_unavailable:" + str(exc))
    try:
        evidence["dirty"] = bool(_git(repo_root, "status", "--porcelain=v1", "--untracked-files=all"))
    except _GitUnavailable as exc:
        reasons.append("dirty_unavailable:" + str(exc))
    evidence["unavailable_reason"] = ";".join(reasons) or None
    return evidence


def collect_runtime() -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Distribution versions describe this Python environment, not a Rust runtime."""
    warnings = []
    packages = {}
    for name in ("numpy", "torch"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
            warnings.append({"field": f"runtime.packages.{name}", "reason": "not_installed"})
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "packages": packages,
    }, warnings


def artifact_record(role: str, reference: PathRef, roots: LocalRoots) -> dict[str, Any]:
    """Hash one explicitly registered file; never enumerate a directory."""
    if not isinstance(role, str) or not role.strip():
        raise ValueError("artifact.role must be a nonempty string")
    path = roots.bind(reference)
    if not path.is_file():
        raise ValueError(f"artifact {reference.root}:{reference.path} must be a readable file")
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                size += len(chunk)
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"artifact {reference.root}:{reference.path} must be a readable file") from exc
    return {"role": role, "root": reference.root, "path": reference.path,
            "size_bytes": size, "sha256": digest.hexdigest()}


def build_manifest(
    resolved: ResolvedConfig,
    *,
    run_id: str,
    roots: LocalRoots,
    artifacts: Iterable[tuple[str, PathRef]] = (),
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    git_collector: Callable[[Path], dict[str, Any]] = collect_git,
    runtime_collector: Callable[[], tuple[dict[str, Any], list[dict[str, str]]]] = collect_runtime,
) -> dict[str, Any]:
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("run_id must contain only letters, digits, '.', '_' or '-' and start alphanumeric")
    captured = clock()
    if captured.tzinfo is None or captured.utcoffset() is None:
        raise ValueError("captured_at_utc requires a timezone-aware clock")
    git = git_collector(roots.repo)
    runtime, warnings = runtime_collector()
    warnings = list(warnings)
    if git["unavailable_reason"]:
        warnings.append({"field": "git", "reason": git["unavailable_reason"]})
    records = [artifact_record(role, ref, roots) for role, ref in artifacts]
    records.sort(key=lambda item: (item["root"], item["path"], item["role"]))
    return {
        "schema_version": 1, "run_id": run_id, "phase": "prepared",
        "captured_at_utc": captured.astimezone(timezone.utc).isoformat(),
        "resolved_config": resolved.to_dict(), "git": git, "runtime": runtime,
        "artifacts": records,
        "collection_warnings": sorted(warnings, key=lambda item: (item["field"], item["reason"])),
    }
