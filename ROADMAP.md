# Roadmap

This roadmap is intentionally conservative. Talibus is a research prototype,
not a production poker bot or live-play assistant.

## v0.1 Research Snapshot

- Current Rust/Python architecture documented.
- Public result pack documented and linked.
- Best-ring ONNX strategy and advantage models released with checksums.
- Setup and smoke-check commands available.
- Limitations and responsible-use boundaries published.
- No large raw buffers, private logs, or PyTorch checkpoints committed.

## v0.2 Reproducible Pipeline — Implemented

- Versioned experiment configuration and portable preparation provenance.
- Deterministic fresh per-network initialization.
- Real CPU traversal, PyTorch training, ONNX export, and Rust evaluation through
  one bounded smoke command.
- Completed/failed execution manifests, seeds, stage status, runtime evidence,
  artifact hashes, and structured results.
- Python/Rust CI and focused configuration, initialization, provenance, and
  orchestration tests.
- Verified smoke runtime/disk observations and honest repeatability limits.
- Existing released model and result evidence preserved.

See [CPU Smoke Experiment](docs/smoke.md) and the
[v0.2 release notes](docs/release-notes-v0.2.md). The reproducible CPU pipeline
is implemented on `main`; future milestones below are optional and deferred.

## v0.3 Evaluation Improvements

- Add confidence intervals where result files provide enough data.
- Expand scripted baseline policy coverage.
- Explore duplicate-match or other variance-reduction experiments where
  appropriate.
- Separate regression metrics from showcase metrics more clearly.

## v0.4 Research Extensions

- Run better abstraction experiments.
- Compare search budgets under consistent evaluation settings.
- Improve model and checkpoint tracking metadata.
- Add clearer experiment manifests for long runs.
- Investigate additional imperfect-information game abstractions without
  overclaiming general strength.
