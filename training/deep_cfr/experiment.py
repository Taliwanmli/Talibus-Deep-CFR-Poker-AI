"""Strict core experiment intent and preparation, without workload orchestration."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from provenance import build_manifest, canonical_json


def _object(value: Any, field: str, required: set[str], optional: set[str] | None = None) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    missing = required - value.keys()
    unknown = value.keys() - required - (optional or set())
    if missing:
        raise ValueError(f"{field}: missing fields {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"{field}: unknown fields {', '.join(sorted(unknown))}")
    return value


def _int(value: Any, field: str, low: int = 1, high: int | None = None) -> int:
    if type(value) is not int or value < low or (high is not None and value > high):
        raise ValueError(f"{field} must be an integer in [{low}, {high if high is not None else 'unbounded'}]")
    return value


def _float(value: Any, field: str, *, zero: bool = False) -> float:
    if type(value) not in (float, int):
        raise ValueError(f"{field} must be a finite number")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{field} must be finite and {'>= 0' if zero else '> 0'}")
    return value


@dataclass(frozen=True)
class PathRef:
    root: str
    path: str

    def __post_init__(self) -> None:
        if self.root not in ("repo", "run"):
            raise ValueError("path.root must be 'repo' or 'run'")
        text = self.path
        if not isinstance(text, str) or not text or PureWindowsPath(text).anchor or text.startswith("/"):
            raise ValueError("path.path must be a nonempty relative portable path")
        if any(char in text for char in '\\~$<>:"|?*') or any(ord(c) < 32 for c in text):
            raise ValueError("path.path contains nonportable characters")
        parts = text.split("/")
        for part in parts:
            if part == "..":
                raise ValueError("path.path must not contain parent traversal")
            if part in ("", "."):
                continue
            if part.endswith((" ", ".")) or re.fullmatch(
                r"(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", part
            ):
                raise ValueError("path.path contains a nonportable Windows component")
        object.__setattr__(self, "path", PurePosixPath(text).as_posix())

    @classmethod
    def parse(cls, value: Any, field: str) -> PathRef:
        obj = _object(value, field, {"root", "path"})
        try:
            return cls(**obj)
        except ValueError as exc:
            raise ValueError(f"{field}: {exc}") from exc


@dataclass(frozen=True)
class LocalRoots:
    """Local bindings only: never embed these absolute paths in serialized data."""
    repo: Path
    run: Path

    def __post_init__(self) -> None:
        for name in ("repo", "run"):
            path = Path(getattr(self, name))
            if not path.is_absolute():
                raise ValueError(f"{name} root binding must be absolute")
            object.__setattr__(self, name, path.resolve())

    def bind(self, reference: PathRef) -> Path:
        root = getattr(self, reference.root)
        path = (root / reference.path).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"path {reference.root}:{reference.path} escapes its root")
        return path


@dataclass(frozen=True)
class GameConfig:
    num_players: int
    starting_stack: int
    small_blind: int
    big_blind: int


@dataclass(frozen=True)
class RunConfig:
    iterations: int
    strategy_every: int


@dataclass(frozen=True)
class ModelSettings:
    hidden_dim: int | None = None
    bottleneck_dim: int | None = None
    dropout_p: float | None = None


@dataclass(frozen=True)
class TraversalConfig:
    traversals: int
    deck_samples: int
    workers: int
    progress_batch: int
    seat_chunks: int
    consolidate_processes: bool


@dataclass(frozen=True)
class TrainingConfig:
    training_steps: int
    batch_size: int
    lr: float
    weight_decay: float
    buffer_size: int
    max_sample_reuse_per_iter: float
    adv_huber_delta: float


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    name: str
    seed: int
    game: GameConfig
    run: RunConfig
    model: ModelSettings
    traversal: TraversalConfig
    training: TrainingConfig
    device: str
    onnx_opset: int
    cluster_dir: PathRef
    output_dir: PathRef

    def to_dict(self) -> dict[str, Any]:
        obj = asdict(self)
        obj["model"] = {k: v for k, v in obj["model"].items() if v is not None}
        obj["execution"] = {"device": obj.pop("device")}
        obj["export"] = {"onnx_opset": obj.pop("onnx_opset")}
        obj["paths"] = {key: obj.pop(key) for key in ("cluster_dir", "output_dir")}
        return obj


def parse_config(value: Any) -> ExperimentConfig:
    obj = _object(value, "config", {
        "schema_version", "name", "seed", "game", "run", "model", "traversal",
        "training", "execution", "export", "paths",
    })
    _int(obj["schema_version"], "schema_version", 1, 1)
    name = obj["name"]
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("name must be an alphanumeric-led label using letters, digits, '.', '_' or '-'")
    seed = _int(obj["seed"], "seed", 0, 2**32 - 1)
    groups = {}
    for key, cls in (("game", GameConfig), ("run", RunConfig),
                     ("traversal", TraversalConfig), ("training", TrainingConfig)):
        groups[key] = dict(_object(obj[key], key, set(cls.__dataclass_fields__)))
    for key in ("game", "run", "traversal", "training"):
        for field, value in groups[key].items():
            label = f"{key}.{field}"
            if field == "consolidate_processes":
                if type(value) is not bool:
                    raise ValueError(f"{label} must be a boolean")
            elif field in ("lr", "weight_decay", "max_sample_reuse_per_iter", "adv_huber_delta"):
                groups[key][field] = _float(value, label, zero=field == "weight_decay")
            else:
                _int(value, label, low=0 if field in ("workers", "strategy_every") else 1)
    game = GameConfig(**groups["game"])
    _int(game.num_players, "game.num_players", 2, 6)
    if not game.small_blind <= game.big_blind <= game.starting_stack:
        raise ValueError("game requires small_blind <= big_blind <= starting_stack")
    traversal = TraversalConfig(**groups["traversal"])
    if traversal.seat_chunks > traversal.traversals:
        raise ValueError("traversal.seat_chunks must not exceed traversals")
    if traversal.consolidate_processes and (traversal.seat_chunks != 1 or game.num_players == 2):
        raise ValueError("traversal.consolidate_processes requires multiseat game and seat_chunks=1")
    model = dict(_object(obj["model"], "model", set(), set(ModelSettings.__dataclass_fields__)))
    for field, value in model.items():
        if field == "dropout_p":
            model[field] = _float(value, "model.dropout_p", zero=True)
            if model[field] >= 1:
                raise ValueError("model.dropout_p must be < 1")
        else:
            _int(value, f"model.{field}")
    device = _object(obj["execution"], "execution", {"device"})["device"]
    if device not in ("cpu", "cuda", "auto"):
        raise ValueError("execution.device must be cpu, cuda or auto")
    opset = _object(obj["export"], "export", {"onnx_opset"})["onnx_opset"]
    _int(opset, "export.onnx_opset")
    paths = _object(obj["paths"], "paths", {"cluster_dir", "output_dir"})
    refs = {key: PathRef.parse(value, f"paths.{key}") for key, value in paths.items()}
    if refs["output_dir"].root != "run":
        raise ValueError("paths.output_dir.root must be run")
    return ExperimentConfig(1, name, seed, game, RunConfig(**groups["run"]),
                            ModelSettings(**model), traversal, TrainingConfig(**groups["training"]),
                            device, opset, **refs)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"duplicate JSON field: {key}")
        obj[key] = value
    return obj


def load_config(path: Path) -> ExperimentConfig:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError("config must be a readable UTF-8 JSON file") from exc
    try:
        obj = json.loads(text, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as exc:
        raise ValueError(f"config: malformed JSON at line {exc.lineno}, column {exc.colno}") from exc
    return parse_config(obj)


@dataclass(frozen=True)
class ResolvedConfig:
    intent: ExperimentConfig
    hidden_dim: int
    bottleneck_dim: int
    dropout_p: float
    input_dim: int
    max_actions: int
    selected_device: str

    def to_dict(self) -> dict[str, Any]:
        obj = self.intent.to_dict()
        obj["model"] = {key: getattr(self, key) for key in (
            "hidden_dim", "bottleneck_dim", "dropout_p", "input_dim", "max_actions",
        )}
        obj["execution"] = {"requested_device": self.intent.device,
                            "selected_device": self.selected_device}
        obj["traversal"]["requested_workers"] = obj["traversal"].pop("workers")
        return obj


def resolve_config(config: ExperimentConfig, *, seed: int | None = None,
                   output_dir: PathRef | None = None) -> ResolvedConfig:
    """Resolve only preparation-time values; no actual worker count is inferred."""
    from model import INPUT_DIM, MAX_ACTIONS, ModelConfig
    from train import resolve_device

    # Revalidate even programmatically constructed objects and every override.
    parse_config(config.to_dict())
    config = replace(config, seed=config.seed if seed is None else seed,
                     output_dir=config.output_dir if output_dir is None else output_dir)
    config = parse_config(config.to_dict())
    defaults = ModelConfig()
    model = {key: getattr(config.model, key) if getattr(config.model, key) is not None
             else getattr(defaults, key) for key in ModelSettings.__dataclass_fields__}
    return ResolvedConfig(config, **model, input_dim=INPUT_DIM, max_actions=MAX_ACTIONS,
                          selected_device=resolve_device(config.device).type)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare experiment provenance JSON; execute no workload.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", help="Override the portable run-relative output path")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--run-root", type=Path, help="Local artifact root; may be outside the checkout")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        override = PathRef("run", args.output_dir) if args.output_dir is not None else None
        resolved = resolve_config(config, seed=args.seed, output_dir=override)
        repo = args.repo_root.resolve()
        roots = LocalRoots(repo, args.run_root.resolve() if args.run_root is not None
                           else repo / "data" / "experiments")
        roots.bind(resolved.intent.cluster_dir)
        roots.bind(resolved.intent.output_dir)
        manifest = build_manifest(resolved, run_id=args.run_id, roots=roots)
        text = canonical_json(manifest)
        # Preserve UTF-8/LF bytes even on Windows; StringIO remains usable by callers.
        if hasattr(sys.stdout, "buffer"):
            sys.stdout.buffer.write(text.encode("utf-8"))
        else:
            sys.stdout.write(text)
    except ValueError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
