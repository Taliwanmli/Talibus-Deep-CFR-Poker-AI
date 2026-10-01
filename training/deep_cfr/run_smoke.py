"""Bounded CPU integration check using the existing training and Rust runtime CLIs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from experiment import LocalRoots, PathRef, ResolvedConfig, load_config, resolve_config
from provenance import artifact_record, build_manifest, canonical_json
from subprocess_env import build_subprocess_env

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("experiments") / "cpu_smoke.json"
MAX_RUN_BYTES = 128 * 1024 * 1024
STAGE_TIMEOUT_SECONDS = 600
BUILD_TIMEOUT_SECONDS = 1200


def validate_smoke(resolved: ResolvedConfig) -> None:
    """This command is deliberately a tiny integration check, not a research runner."""
    c = resolved.intent
    if c.device != "cpu" or resolved.selected_device != "cpu":
        raise ValueError("CPU smoke requires execution.device=cpu")
    if c.game.num_players != 6 or c.run.iterations != 1 or c.run.strategy_every != 1:
        raise ValueError("CPU smoke requires six players, one iteration and strategy_every=1")
    if c.game.starting_stack > 3 * c.game.big_blind:
        raise ValueError("CPU smoke requires starting_stack <= 3 big blinds")
    if c.traversal.workers != 1 or c.traversal.seat_chunks != 1 or not c.traversal.consolidate_processes:
        raise ValueError("CPU smoke requires one traversal worker, one seat chunk and consolidated processes")
    limits = {
        "traversals": (c.traversal.traversals, 8),
        "deck_samples": (c.traversal.deck_samples, 8),
        "hidden_dim": (resolved.hidden_dim, 64),
        "bottleneck_dim": (resolved.bottleneck_dim, 32),
        "training_steps": (c.training.training_steps, 8),
        "batch_size": (c.training.batch_size, 64),
        "buffer_size": (c.training.buffer_size, 4096),
    }
    for field, (value, limit) in limits.items():
        if value > limit:
            raise ValueError(f"CPU smoke {field} must be <= {limit}")
    if c.evaluation is None:
        raise ValueError("CPU smoke requires an explicit evaluation configuration")
    if c.evaluation.workers != 1 or c.evaluation.hands > 100 or c.evaluation.deck_samples > 8:
        raise ValueError("CPU smoke evaluation requires one worker, <=100 hands and <=8 deck samples")


def training_command(resolved: ResolvedConfig, roots: LocalRoots, pipeline: Path,
                     traversal_binary: Path) -> list[str]:
    c = resolved.intent
    values = {
        "iterations": c.run.iterations, "strategy-every": c.run.strategy_every,
        "traversals": c.traversal.traversals, "deck-samples": c.traversal.deck_samples,
        "traversal-workers": c.traversal.workers, "traversal-progress-batch": c.traversal.progress_batch,
        "traversal-seat-chunks": c.traversal.seat_chunks,
        "num-players": c.game.num_players, "starting-stack": c.game.starting_stack,
        "small-blind": c.game.small_blind, "big-blind": c.game.big_blind,
        "hidden-dim": resolved.hidden_dim, "bottleneck-dim": resolved.bottleneck_dim,
        "dropout-p": resolved.dropout_p, "training-steps": c.training.training_steps,
        "batch-size": c.training.batch_size, "lr": c.training.lr,
        "weight-decay": c.training.weight_decay, "buffer-size": c.training.buffer_size,
        "max-sample-reuse-per-iter": c.training.max_sample_reuse_per_iter,
        "adv-huber-delta": c.training.adv_huber_delta, "seed": c.seed,
        "device": resolved.selected_device, "onnx-opset": c.onnx_opset,
        "cluster-dir": roots.bind(c.cluster_dir), "work-dir": pipeline,
        "rust-binary": traversal_binary, "diagnostic-every": 0, "exploitability-every": 0,
        "league-eval-every": 0, "ring-eval-every": 0, "checkpoint-eval-every": 0,
        "buffer-save-every": 1, "model-checkpoint-every": 0, "log-every": 1,
    }
    command = [sys.executable, str(roots.repo / "training/deep_cfr/run_deep_cfr.py")]
    for flag, value in values.items():
        command.extend([f"--{flag}", str(value)])
    command.append("--keep-samples")
    if c.traversal.consolidate_processes:
        command.append("--traversal-consolidate-processes")
    return command


def evaluation_command(resolved: ResolvedConfig, roots: LocalRoots, model: Path,
                       evaluation_binary: Path) -> list[str]:
    c = resolved.intent
    e = c.evaluation
    assert e is not None
    values = {
        "model": model, "policy": "strategy", "mode": "vs-opponent",
        "opponent": e.opponent, "hands": e.hands, "deck-samples": e.deck_samples,
        "workers": e.workers, "seed": e.seed, "model-seat": e.model_seat,
        "num-players": c.game.num_players, "starting-stack": c.game.starting_stack,
        "small-blind": c.game.small_blind, "big-blind": c.game.big_blind,
        "cluster-dir": roots.bind(c.cluster_dir), "progress-every": 1,
    }
    command = [str(evaluation_binary)]
    for flag, value in values.items():
        command.extend([f"--{flag}", str(value)])
    return command


def portable_command(command: list[str], roots: LocalRoots) -> list[str]:
    """Record executable intent without publishing machine-specific absolute paths."""
    result = []
    for value in command:
        if value == sys.executable:
            result.append("python")
        elif Path(value).is_absolute():
            path = Path(value).resolve()
            for name in ("run", "repo"):
                root = getattr(roots, name)
                if path.is_relative_to(root):
                    result.append(f"{name}:{path.relative_to(root).as_posix()}")
                    break
            else:
                result.append("external-executable:" + path.name)
        else:
            result.append(value)
    return result


def run_stage(name: str, command: list[str], cwd: Path, env: dict[str, str],
              output: Path, stages: list[dict[str, Any]], roots: LocalRoots,
              timeout: int = STAGE_TIMEOUT_SECONDS) -> None:
    print(f"[smoke] {name} (log: {output})", flush=True)
    stage = {"name": name, "command": portable_command(command, roots), "status": "running"}
    stages.append(stage)
    started = time.perf_counter()
    with output.open("w", encoding="utf-8", newline="\n") as log:
        try:
            process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=os.name != "nt")
        except OSError as exc:
            stage.update(status="failed", failure="process_launch",
                         elapsed_seconds=round(time.perf_counter() - started, 3))
            raise RuntimeError(f"cannot launch {name}; check required executables and {output}") from exc
        try:
            code = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name != "nt":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()
            stage.update(status="failed", failure="timeout_or_interruption")
            raise RuntimeError(f"{name} exceeded its time budget or was interrupted; inspect {output}")
        finally:
            stage["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    stage.update(status="completed" if code == 0 else "failed", exit_code=code)
    if code:
        raise RuntimeError(f"{name} failed with exit code {code}; inspect {output}")


def inspect_training(pipeline: Path, resolved: ResolvedConfig) -> dict[str, Any]:
    from train import ADVANTAGE_SAMPLE_MAGIC, STRATEGY_SAMPLE_MAGIC, load_binary_samples
    state = json.loads((pipeline / "state.json").read_text(encoding="utf-8"))
    metrics = json.loads((pipeline / "metrics.json").read_text(encoding="utf-8"))
    entries = metrics.get("iterations", [])
    if state.get("completed_iterations") != 1 or len(entries) != 1:
        raise ValueError("training did not record exactly one completed smoke iteration")
    entry = entries[0]
    counts = {}
    for family, magic in (("advantage", ADVANTAGE_SAMPLE_MAGIC), ("strategy", STRATEGY_SAMPLE_MAGIC)):
        stem = "adv" if family == "advantage" else "strategy"
        path = pipeline / "samples" / f"samples_{stem}_iter001.bin"
        features, targets, masks, iterations = load_binary_samples(path, expected_magic=magic)
        count = len(features)
        if count == 0 or count != entry["samples"][f"{family}_merged"]:
            raise ValueError(f"{family} retained samples do not match recorded training input")
        import numpy as np
        if not np.isfinite(features).all() or not np.isfinite(targets).all():
            raise ValueError(f"{family} samples contain non-finite values")
        if not (iterations == 1).all() or not (masks.sum(axis=1) > 0).all():
            raise ValueError(f"{family} samples have invalid iteration or action masks")
        steps = entry["training"][f"{family}_steps_ran"]
        loss = entry["loss"][family]
        if not isinstance(steps, int) or not 0 < steps <= resolved.intent.training.training_steps:
            raise ValueError(f"{family} training did not perform the configured work")
        if not isinstance(loss, (int, float)) or not math.isfinite(loss):
            raise ValueError(f"{family} training recorded a non-finite loss")
        counts[family] = {"samples": count, "steps_ran": steps, "final_loss": loss}
    if not entry["strategy_snapshot"]:
        raise ValueError("the smoke strategy was not trained/exported")
    return {"completed_iterations": 1, "networks": counts, "fresh_work_directory": True,
            "resume": False, "device": metrics["config"]["device_resolved"]}


def inspect_models(pipeline: Path, resolved: ResolvedConfig) -> dict[str, Any]:
    import onnx
    import torch
    from model import DeepCfrNet, ModelConfig, INPUT_DIM, MAX_ACTIONS
    evidence = {}
    for family, salt in (("advantage", 0x9E37_79B9), ("strategy", 0x94D0_49BB)):
        path = pipeline / "models" / f"{family}_shared"
        checkpoint = torch.load(path.with_suffix(".pt"), map_location="cpu", weights_only=True)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(resolved.intent.seed ^ salt)
            initial = DeepCfrNet(ModelConfig(hidden_dim=resolved.hidden_dim,
                                bottleneck_dim=resolved.bottleneck_dim, dropout_p=resolved.dropout_p))
        weights = checkpoint["state_dict"]
        if checkpoint["network_type"] != family or not all(torch.isfinite(t).all() for t in weights.values()):
            raise ValueError(f"invalid smoke {family} checkpoint")
        changed = [name for name, value in initial.state_dict().items() if not torch.equal(value, weights[name])]
        if not changed:
            raise ValueError(f"{family} checkpoint still matches fresh untrained initialization")
        model = onnx.load(str(path.with_suffix(".onnx")))
        onnx.checker.check_model(model)
        tensors = list(model.graph.input) + list(model.graph.output)
        if any(item.type.tensor_type.elem_type != onnx.TensorProto.FLOAT for item in tensors):
            raise ValueError(f"{family} ONNX input/output tensors must be float32")
        shapes = {item.name: [d.dim_value or d.dim_param for d in item.type.tensor_type.shape.dim]
                  for item in tensors}
        expected = {"input": ["batch", INPUT_DIM], "action_mask": ["batch", MAX_ACTIONS],
                    "output": ["batch", MAX_ACTIONS]}
        if shapes != expected:
            raise ValueError(f"{family} ONNX input/output contract does not match Talibus runtime")
        evidence[family] = {"initialization_seed": resolved.intent.seed ^ salt,
                            "changed_parameter_tensors": len(changed), "onnx_shapes": shapes}
    return evidence


def parse_evaluation(log: str, resolved: ResolvedConfig) -> dict[str, Any]:
    matches = [line.removeprefix("RING_EVAL_JSON ") for line in log.splitlines()
               if line.startswith("RING_EVAL_JSON ")]
    if len(matches) != 1:
        raise ValueError("Rust evaluation must emit exactly one RING_EVAL_JSON result")
    result = json.loads(matches[0])
    c = resolved.intent
    e = c.evaluation
    assert e is not None
    for name, expected in (("hands", e.hands), ("num_players", c.game.num_players),
                           ("model_seat", e.model_seat), ("opponent", e.opponent.replace("-", "_")),
                           ("mode", "vs_opponent")):
        if result.get(name) != expected:
            raise ValueError(f"Rust evaluation {name} disagrees with configuration")
    outcomes = [result[key] for key in ("model_wins", "model_losses", "ties")]
    if any(type(value) is not int or value < 0 for value in outcomes) or sum(outcomes) != e.hands:
        raise ValueError("Rust evaluation outcome counts do not match completed hands")
    for name in ("per_seat_bb_per_hand", "per_seat_bb_per_100"):
        if len(result[name]) != c.game.num_players or not all(math.isfinite(value) for value in result[name]):
            raise ValueError("Rust evaluation returned invalid per-seat utilities")
    for name in ("model_bb_per_hand", "model_bb_per_100", "elapsed_sec", "zero_sum_check_bb_per_hand"):
        if not math.isfinite(result[name]):
            raise ValueError("Rust evaluation returned a non-finite scalar")
    if abs(result["zero_sum_check_bb_per_hand"]) > 1e-6:
        raise ValueError("Rust evaluation failed the simulator zero-sum check")
    return {"purpose": "pipeline-health only; not a poker-strength benchmark", "seed": e.seed,
            "runtime": "Rust OnnxPolicy CPU inference", "result": result}


def record_runtime(env: dict[str, str]) -> dict[str, Any]:
    packages = {name: metadata.version(name) for name in
                ("torch", "numpy", "onnx", "onnxruntime", "onnxscript", "onnx_ir")}
    rust = subprocess.run(["rustc", "--version"], capture_output=True, text=True, check=True, timeout=10)
    cargo = subprocess.run(["cargo", "--version"], capture_output=True, text=True, check=True, timeout=10)
    path = env.get("ORT_DYLIB_PATH")
    if not path or not Path(path).is_file():
        raise ValueError("ONNX Runtime shared library unavailable; install onnxruntime>=1.23 or set ORT_DYLIB_PATH")
    library = Path(path)
    digest = hashlib.sha256(library.read_bytes()).hexdigest()
    return {"packages": packages, "rustc": rust.stdout.strip(), "cargo": cargo.stdout.strip(),
            "onnxruntime_library": {"filename": library.name, "sha256": digest,
                                    "size_bytes": library.stat().st_size},
            "cpu_threads": 1, "cuda_visible_devices": "", "execution_provider": "CPU"}


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json(value).encode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    parser.add_argument("--run-root", type=Path, default=REPO_ROOT / "data/experiments")
    args = parser.parse_args(argv)
    destination = None
    manifest = None
    stages: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        config = load_config(args.config)
        resolved = resolve_config(config, output_dir=PathRef("run", f"{config.output_dir.path}/{args.run_id}"))
        validate_smoke(resolved)
        roots = LocalRoots(REPO_ROOT, args.run_root.resolve())
        cluster = roots.bind(config.cluster_dir)
        if not cluster.is_dir():
            raise ValueError("configured abstraction cluster directory is missing")
        manifest = build_manifest(resolved, run_id=args.run_id, roots=roots)
        candidate = roots.bind(resolved.intent.output_dir)
        candidate.mkdir(parents=True, exist_ok=False)
        destination = candidate
        write_json(destination / "resolved_config.json", resolved.to_dict())
        write_json(destination / "manifest.json", manifest)
        env = build_subprocess_env()
        env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        env.pop("TALIBUS_DEBUG_LOG", None)
        env["PYTHONHASHSEED"] = str(config.seed)
        runtime = record_runtime(env)
        suffix = ".exe" if os.name == "nt" else ""
        target = Path(env.get("CARGO_TARGET_DIR", str(roots.repo / "solver/target")))
        if not target.is_absolute():
            target = roots.repo / "solver" / target
        binaries = {name: target / "release" / (name + suffix) for name in ("run_traversals", "ring_game_eval")}
        build = ["cargo", "build", "--release", "--locked", "-p", "deep_cfr",
                 "--bin", "run_traversals", "--bin", "ring_game_eval"]
        run_stage("build Rust traversal and evaluation", build, roots.repo / "solver", env,
                  destination / "build.log", stages, roots, BUILD_TIMEOUT_SECONDS)
        pipeline = destination / "pipeline"
        run_stage("generate samples, train fresh networks and export ONNX",
                  training_command(resolved, roots, pipeline, binaries["run_traversals"]), roots.repo, env,
                  destination / "training.log", stages, roots)
        training = inspect_training(pipeline, resolved)
        models = inspect_models(pipeline, resolved)
        run_stage("load smoke ONNX and evaluate in Rust",
                  evaluation_command(resolved, roots, pipeline / "models/strategy_shared.onnx", binaries["ring_game_eval"]),
                  roots.repo, env, destination / "evaluation.log", stages, roots)
        evaluation = parse_evaluation((destination / "evaluation.log").read_text(encoding="utf-8"), resolved)
        write_json(destination / "evaluation.json", evaluation)
        summary = {"training": training, "models": models, "evaluation": evaluation,
                   "seeds": {"base": config.seed, "traversal": config.seed ^ 0x9E37_79B9 ^ (6 * 0x94D0_49BB),
                             "advantage_training": config.seed ^ 0xF135_7AEA,
                             "strategy_ingestion": config.seed ^ 0x0B7F_4D95,
                             "strategy_training": config.seed ^ 0x5F35_9ACD, "evaluation": config.evaluation.seed},
                   "determinism": "Seeds and one-worker CPU execution are controlled; compare artifacts on the same software/platform. No cross-platform bitwise guarantee; timestamps/timings differ."}
        write_json(destination / "summary.json", summary)
        artifacts = []
        for path in sorted(destination.rglob("*")):
            if path.is_file() and path.name != "manifest.json":
                ref = PathRef("run", path.relative_to(roots.run).as_posix())
                artifacts.append(("generated_" + path.suffix.removeprefix("."), ref))
        for path in sorted(cluster.iterdir()):
            if path.is_file():
                artifacts.append(("abstraction_asset", PathRef(config.cluster_dir.root,
                                 (Path(config.cluster_dir.path) / path.name).as_posix())))
        artifacts.append(("rust_dependency_lock", PathRef("repo", "solver/Cargo.lock")))
        records = [artifact_record(role, ref, roots) for role, ref in artifacts]
        size = sum(path.stat().st_size for path in destination.rglob("*") if path.is_file())
        if size > MAX_RUN_BYTES:
            raise ValueError("smoke outputs exceeded the 128 MiB artifact budget")
        manifest.update(phase="completed", artifacts=records,
                        execution={"stages": stages, "runtime": runtime, "summary": summary,
                                   "elapsed_seconds": round(time.perf_counter() - started, 3),
                                   "generated_bytes_before_final_manifest": size})
        write_json(destination / "manifest.json", manifest)
        print(f"[smoke] COMPLETE: {destination}\n[smoke] manifest: {destination / 'manifest.json'}\n"
              f"[smoke] results: {destination / 'evaluation.json'}", flush=True)
        return 0
    except Exception as exc:
        if destination is not None and manifest is not None and destination.is_dir():
            manifest.update(phase="failed", execution={"stages": stages, "failure_type": type(exc).__name__})
            write_json(destination / "manifest.json", manifest)
        message = str(exc)
        if isinstance(exc, (ImportError, metadata.PackageNotFoundError)):
            message += "; install dependencies with: python -m pip install -r training/deep_cfr/requirements.txt"
        print(f"[smoke] FAILED: {message}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
