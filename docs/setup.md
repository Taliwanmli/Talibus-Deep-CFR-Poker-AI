# Setup

This document describes the development setup for Talibus.

## Prerequisites

- Rust stable with Cargo.
- Python 3.10 or newer.
- A C/C++ toolchain suitable for Rust native dependencies.
- Enough disk space for generated `data/` artifacts if running training.
- Optional CUDA-capable GPU for larger PyTorch training runs.

The Rust runtime uses `ort` 2.0.0-rc.11 for ONNX inference. Use ONNX Runtime
1.23.x or newer, and make the shared library available to the binaries if it
is not on the platform library path. For example, set `ORT_DYLIB_PATH` to the
ONNX Runtime shared library before running binaries that load `.onnx` models.

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

From `solver/`:

```bash
cargo check --workspace
cargo test -p cfr
cargo test -p abstraction
cargo build --release -p deep_cfr --bin run_traversals
cargo build --release -p deep_cfr --bin ring_game_eval
cargo build --release -p deep_cfr --bin realtime_play
```

## Smoke Checks

The repository includes a trained ONNX strategy model under
`artifacts/models/talibus-6max-longrun-opt-v1/`. Full long-run
training/evaluation still requires generated local artifacts and substantial
compute, but lightweight source and model smoke checks are available.

The checks below are the closest lightweight verification path. They verify
that key Python modules parse, command-line entry points are discoverable, and
the fast evaluation tests pass without launching training jobs or creating
large artifacts.

From the repository root, after installing the Python requirements:

```bash
python3 -m py_compile \
  training/deep_cfr/model.py \
  training/deep_cfr/train.py \
  training/deep_cfr/reservoir.py \
  training/deep_cfr/run_deep_cfr.py \
  run_eval_suite.py
```

Verifies that the selected training/evaluation Python files parse correctly.

```bash
python3 run_eval_suite.py --help
```

Verifies that the structured evaluation-suite CLI is importable and exposes its
options without requiring a model artifact.

```bash
python3 -m eval.run_league --help
```

Verifies that the offline evaluation CLI is importable and exposes its options.
Some environments may print a Gym deprecation warning; the smoke check still
passes if the command exits successfully.

```bash
python3 -m unittest discover eval
```

Runs the fast Python evaluation tests. These tests do not require trained ONNX
models or long generated training buffers.

On systems where `python` points to Python 3, use `python` instead of
`python3`.

Optional Rust verification, if Cargo is installed:

```bash
cd solver
cargo check --workspace
```

This checks the Rust workspace without running training or evaluation. It may
take longer on a first build because Cargo has to download and compile
dependencies.

## Training Entry Point

The main training orchestrator is:

```bash
python training/deep_cfr/run_deep_cfr.py --help
```

A full 6-max run requires compiled Rust traversal binaries, cluster assets,
and a writable `data/` directory. Use small smoke settings first before
starting a long run.

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
end-to-end smoke experiment. Existing training/evaluation CLIs do not yet consume
the new configuration.

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
