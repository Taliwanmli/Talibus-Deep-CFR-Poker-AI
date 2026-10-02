# Limitations

Talibus is a research prototype with important limitations.

- The game uses abstraction. Action choices are mapped into fixed slots and
  postflop states use card/board clustering.
- Large training buffers, raw logs, and PyTorch checkpoints are not tracked in
  Git.
- The released ONNX artefacts are the compact best-ring strategy/advantage
  models under `artifacts/models/talibus-6max-longrun-opt-v1/`. Other local
  generated artefacts are intentionally excluded.
- The included compact result pack records simulator measurements, model
  artifact hashes, and benchmark settings.
- Scripted-opponent win rates do not imply performance against strong human
  players or external solver systems.
- This project does not prove Deep CFR convergence in multiplayer poker.
- Some orchestration scripts were developed for a specific local/Windows
  workflow and may need environment-specific adjustment.
- Depth-limited search timings depend heavily on hardware, ONNX Runtime setup,
  thread count, and batch settings.
- The CPU smoke validates traversal/training/export/Rust integration with a
  tiny model and short-stack game; its losses and evaluation utilities are not
  model-strength evidence.
- Same-environment smoke repeats have been checked, but cross-platform/version
  bitwise determinism is not guaranteed. Requirements use lower bounds and
  manifests record the installed versions.
- The historical result pack does not preserve the original terminal
  transcript, exact source commit, confidence intervals, or full training
  inputs. The new smoke does not reconstruct that long run.

The intended public framing is a Deep-CFR-style 6-max NLHE research prototype,
not a production poker bot.
