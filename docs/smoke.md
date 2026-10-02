# CPU Smoke Experiment

This experiment verifies Talibus's real Rust/Python/ONNX integration on CPU.
It generates new samples and trains new models; it does not use the historical
released model or long-run training buffers. The result is pipeline-health
evidence, not a poker-strength benchmark.

## Run

Install Rust stable, a native build toolchain, and both Python requirement files
as described in [Setup](setup.md#python-environment). From the repository root:

```bash
python training/deep_cfr/run_smoke.py
```

The command builds `run_traversals` and `ring_game_eval` in release mode with
`Cargo.lock`, runs the existing `run_deep_cfr.py` training orchestrator, then
loads the exported strategy ONNX through the existing Rust `OnnxPolicy` and
model-only ring evaluation. Stage progress and log locations are printed.
Success prints `COMPLETE`, the manifest path, and the result path, and exits 0.

The wrapper discovers the native ONNX Runtime library bundled with the active
Python `onnxruntime` package. A separate runtime download is normally
unnecessary. If the bundled library is unavailable, use `ORT_DYLIB_PATH` to
select a compatible library. CUDA is hidden from child processes and training
explicitly selects CPU.

Every default invocation gets a new timestamp-based run ID. An explicit ID is
also supported:

```bash
python training/deep_cfr/run_smoke.py --run-id reviewer-check
```

Existing run directories are rejected rather than reused. To keep outputs
outside the checkout, pass `--run-root <local-artifact-root>`. `--config` selects
another input using the same experiment contract, subject to the smoke bounds.
The wrapper is deliberately limited to small integration runs.

## What Executes

The checked-in [cpu_smoke.json](../training/deep_cfr/experiments/cpu_smoke.json)
is resolved by `experiment.py` and then consumed by the wrapper:

| Stage | Default settings / evidence |
| --- | --- |
| Game and traversal | Six players, 60-chip stacks, 10/20 blinds; two traversals per player, two deck samples, one worker, consolidated traversal. |
| Fresh training | One iteration; 32/16 hidden/bottleneck widths, 64-sample reservoirs, batch size 8, two optimizer steps for each network. |
| Export | Newly trained advantage and strategy models, ONNX opset 17. |
| Rust inference/evaluation | Newly exported strategy model; 12 hands against random scripted policies, one worker, evaluation seed 43. |
| Recording | Resolved configuration, Git/runtime evidence, derived seeds, executed commands/status, structured results, sizes and SHA-256 hashes. |

The wrapper checks that samples are present and finite, training actually ran,
checkpoint tensors changed from fresh initialization, and both ONNX models
have the expected float32 inputs/output: `[batch, 510]`, `[batch, 13]`, and
`[batch, 13]`. Rust must complete the configured evaluation and emit a result
with valid outcomes and the simulator zero-sum check. A failed check makes the
command fail.

This profile's short stacks and tiny networks reduce integration cost. They do
not match the historical model's long-run training/evaluation settings.
Depth-limited search is an additional research runtime path; the CPU smoke
uses model-only evaluation.

## Outputs And Inspection

Outputs live under `data/experiments/cpu-smoke/<run-id>/` by default:

```text
manifest.json                 Completed or failed execution evidence
resolved_config.json          Exact validated settings
summary.json                  Sample counts, training steps, models, seeds, results
evaluation.json               Structured Rust model-only evaluation result
build.log / training.log / evaluation.log
pipeline/samples/             Retained binary traversal samples
pipeline/models/              Fresh .pt checkpoints and .onnx exports
pipeline/buffers/              Small local reservoirs
pipeline/state.json / metrics.json
```

Inspect `manifest.json`: success requires `phase: "completed"` and completed
stages with exit code 0. Its `execution.summary` records actual sample counts,
optimizer steps, fresh initialization evidence, tensor shapes, and evaluation
outcomes. `artifacts` identifies generated files, abstraction assets, and
`solver/Cargo.lock` with portable `repo`/`run` references, sizes, and SHA-256.
The manifest does not hash itself.

The run is ignored by Git. Historical files under `artifacts/models/` and
`results/` are left intact. Legacy training metrics and logs can contain local
absolute paths; review those files before sharing a run directory. The wrapper's
resolved settings and manifest references use portable paths.

## Observed Cost

Two complete verification runs on macOS arm64 with cached Rust release builds
took about 4 seconds each (3.857 and 3.599 seconds) and generated about 1.8 MB
per run, including the final manifest. Each produced 82 advantage samples,
468 strategy samples, and two optimizer steps for each network.

A clean checkout with no existing Rust target directory completed in about
14 seconds (13.596 seconds), including about 10 seconds of release compilation.
It generated about 1.8 MB of run output plus a separate 152 MiB Rust release
build directory. Cargo's downloaded crate cache was already available.

These measurements exclude Python dependency installation and Cargo downloads.
Allow additional time and disk space for dependencies and `solver/target/` on
a first setup. Runtime and sample counts can differ with platform/software
changes; these are observed values, not performance guarantees. The wrapper
rejects outputs above 128 MiB after execution and applies stage timeouts.

The measured environment used Python 3.12.13, PyTorch 2.14.1, NumPy 2.5.3,
ONNX 1.23.1, ONNX Runtime 1.30.0, onnxscript 0.7.2, onnx_ir 1.0.0, and
Rust/Cargo 1.98.1. Each manifest records its own environment and the native
ONNX Runtime library filename, size, and hash.

## Reproducibility Boundaries

The base seed controls fresh per-network initialization and the existing
derived traversal, reservoir ingestion, and training seed streams. Evaluation
has its own explicit seed. The smoke fixes one worker and CPU thread limits,
preserves sample files, and records all these choices.

The two measured runs on the same platform/software produced identical binary
sample hashes, ONNX hashes, and checkpoint parameter tensors. Evaluation
results matched apart from elapsed time. A separate clean checkout run from a
different working directory also produced matching sample and ONNX hashes.
This supports repeatability for that environment; it does not establish
bit-for-bit reproducibility across operating systems, dependency versions, or
hardware. The NLHE model uses Rust's `DefaultHasher` for hash-derived behavior;
its algorithm is not a stable cross-version contract. Timestamps, durations,
serialization metadata, and local-path-bearing metrics/logs may differ even
when model parameters match.

Git commit/dirty evidence does not capture uncommitted source bytes. Python
requirements specify supported lower bounds rather than a locked environment;
use recorded versions when investigating a difference. The historical long-run
training inputs/checkpoints are still excluded from Git, and this smoke does
not reproduce the published long-run result pack.

## Failure Handling

A failed subprocess or verification check exits nonzero and identifies the
problem/log. Once a run directory exists, failure records `phase: "failed"`
and available stage status in its manifest. Partial output is retained for
diagnosis; retry with a new run ID after addressing the error. An invalid
configuration or existing destination can fail before a new manifest is
created. Successful runs need no temporary-directory cleanup.

## Automated Checks

Focused orchestration and native-library discovery tests:

```bash
python -m unittest discover -s training/deep_cfr -p "test_smoke.py" -v
python -m unittest discover -s training/deep_cfr -p "test_subprocess_env.py" -v
```

These inexpensive tests support the wrapper. The actual command above is the
end-to-end proof. Normal CI runs source, CLI, and unit checks; the full smoke is
kept separate because it installs export/runtime dependencies and compiles Rust
release binaries. See [Contributing](../CONTRIBUTING.md#basic-checks) for the
Python and Rust CI-equivalent commands.
