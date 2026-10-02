# Setup

This document describes the development setup for Talibus.

## Prerequisites

- Rust stable with Cargo.
- Python 3.10 or newer.
- A C/C++ toolchain suitable for Rust native dependencies.
- Enough disk space for generated `data/` artifacts if running training.
- Optional CUDA-capable GPU for larger PyTorch training runs.

The Rust runtime uses `ort` 2.0.0-rc.11 for ONNX inference. Use a compatible
ONNX Runtime 1.x library (at least 1.23). The CPU smoke wrapper discovers the
native library bundled with Python `onnxruntime` on macOS, Linux, and Windows.
Direct Rust binary invocations require the library on the platform library
path or `ORT_DYLIB_PATH` set to its actual `.dylib`, `.so`, or `.dll` location.

## Python Environment

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r training/deep_cfr/requirements.txt
pip install -r eval/requirements.txt
```

On Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r training\deep_cfr\requirements.txt
python -m pip install -r eval\requirements.txt
```

## Rust Build

The CPU smoke command builds its required Rust binaries automatically. For
manual development checks, run from the repository root:

```bash
cargo check --manifest-path solver/Cargo.toml --workspace --locked
cargo test --manifest-path solver/Cargo.toml --workspace --locked
cargo build --manifest-path solver/Cargo.toml --release --locked -p deep_cfr --bin run_traversals --bin ring_game_eval --bin realtime_play
```

## CPU End-To-End Verification

After installing the Python requirements, run from the repository root:

```bash
python training/deep_cfr/run_smoke.py
```

This is the shortest real pipeline check: traversal/sample generation, fresh
PyTorch training, ONNX export, Rust loading/inference, and structured model-only
evaluation. No CUDA, historical buffers, or released model is required. The
checked-in CPU configuration controls the run. Outputs and a completed manifest
are printed and stored under `data/experiments/cpu-smoke/<run-id>/`.

See [CPU Smoke Experiment](smoke.md) for configuration bounds, observed runtime
and disk use, failure handling, artifacts, and reproducibility guarantees.

## Source And Test Checks

Faster checks from the repository root verify source, CLI discovery, and unit
contracts without executing the complete training/runtime pipeline:

```bash
python -m pip check
python -m compileall -q training/deep_cfr eval run_eval_suite.py
python run_eval_suite.py --help
python -m eval.run_league --help
python training/deep_cfr/run_deep_cfr.py --help
python training/deep_cfr/run_smoke.py --help
python -m unittest discover eval
python -m unittest discover -s training/deep_cfr -p "test_*.py"
```

These tests do not require the released model or long training buffers. Some
evaluation commands may print upstream dependency warnings; use exit status to
determine success. [Contributing](../CONTRIBUTING.md#basic-checks) lists the Rust
formatting, lint, and test commands used by CI. Dependency downloads and first
compilation take additional time.

## Training Entry Point

The main training orchestrator is:

```bash
python training/deep_cfr/run_deep_cfr.py --help
```

A full 6-max run requires compiled Rust traversal binaries, cluster assets,
and a writable `data/` directory. Use the bounded CPU smoke command first;
long-run reproduction is a separate, substantially larger workflow.

## Preparing An Experiment Manifest

After installing the Python training dependencies, run from the repository root:

```bash
python training/deep_cfr/experiment.py --config training/deep_cfr/experiments/example.json --run-id example-preparation
```

This prints a versioned JSON manifest to stdout. It validates the input, expands
supported model defaults/constants, selects a device using the existing device
resolver, and collects Git and Python environment evidence. The small example
explicitly requests CPU and one worker. `phase` is always `prepared` and
`artifacts` is empty for this command.

The command creates no output directory or artifact. It does not launch
traversal, train a model, export ONNX, load Rust runtime models, or evaluate
poker. It requires neither compiled Rust binaries nor CUDA. The example's small
budgets are illustrative settings, not evidence of training quality or a tested
end-to-end smoke experiment. `run_smoke.py` consumes this contract for the real
CPU workflow and adds executed-stage evidence to a completed/failed manifest.
General long-run training/evaluation CLIs retain their existing flag interfaces.

Optional overrides are `--seed` and `--output-dir` (a portable run-relative
path). `--run-root` binds local artifact storage, including storage outside the
checkout; it defaults to `data/experiments` under the repository. `--repo-root`
defaults to the checkout inferred from the script location. Neither absolute
binding is serialized. For example, supply `--run-root "<local-artifact-root>"`
using your actual local directory without editing the JSON configuration.

The example's output reference is `run:example`, relative to that storage root.
Automatic worker selection is also representable with `workers: 0`; the manifest
retains it as `requested_workers: 0`, without predicting an actual worker count.
An unavailable Git checkout yields explicit null provenance and a reason.

To write the exact canonical UTF-8 bytes independently of shell redirection
encoding, capture stdout with Python. Run this from the repository root:

```python
import subprocess
import sys
from pathlib import Path

result = subprocess.run(
    [sys.executable, "training/deep_cfr/experiment.py", "--config",
     "training/deep_cfr/experiments/example.json", "--run-id", "example-preparation"],
    check=True, capture_output=True,
)
destination = Path("data/experiments/example/manifest.json")
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_bytes(result.stdout)
```

This captures provenance before creating the output file/directory. The caller
chooses whether and where to save the manifest.

Focused checks:

```bash
python -m unittest discover -s training/deep_cfr -p "test_experiment.py" -v
python -m unittest discover -s training/deep_cfr -p "test_provenance.py" -v
```

See [the architecture contract](architecture.md#experiment-configuration-and-preparation-provenance)
for field meanings, root bindings, unavailable evidence, and determinism limits.

## Evaluation Entry Point

The main evaluation suite is:

```bash
python run_eval_suite.py --help
```

The suite expects a trained ONNX model and compiled `ring_game_eval` /
`realtime_play` binaries. The released strategy model is:

```text
artifacts/models/talibus-6max-longrun-opt-v1/strategy_shared_best_ring.onnx
```

After building the Rust binaries, run a short scripted-opponent model smoke
test from the repository root:

```bash
export ORT_DYLIB_PATH=/path/to/libonnxruntime.so
solver/target/release/ring_game_eval \
  --model artifacts/models/talibus-6max-longrun-opt-v1/strategy_shared_best_ring.onnx \
  --policy strategy \
  --cluster-dir checkpoints/nlhe_clusters \
  --num-players 6 \
  --hands 100 \
  --opponent tag
```

On Windows PowerShell, set `ORT_DYLIB_PATH` to the compatible
`onnxruntime.dll` if the DLL is not already discoverable:

```powershell
$env:ORT_DYLIB_PATH = "<path-to-onnxruntime.dll>"
.\solver\target\release\ring_game_eval.exe `
  --model .\artifacts\models\talibus-6max-longrun-opt-v1\strategy_shared_best_ring.onnx `
  --policy strategy `
  --cluster-dir .\checkpoints\nlhe_clusters `
  --num-players 6 `
  --hands 100 `
  --opponent tag
```

## Optional Debug Logs

By default, Talibus does not write debug logs. To enable best-effort JSONL
diagnostics, set:

```bash
export TALIBUS_DEBUG_LOG=debug/talibus_debug.jsonl
```

On Windows PowerShell:

```powershell
$env:TALIBUS_DEBUG_LOG = "debug\talibus_debug.jsonl"
```
