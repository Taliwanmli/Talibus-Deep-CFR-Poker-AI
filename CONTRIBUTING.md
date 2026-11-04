# Contributing

Talibus is a research prototype for imperfect-information game AI and systems
engineering. Contributions should preserve that scope.

## Project Scope

Appropriate contributions improve documentation, reproducibility, simulator
evaluation, tests, training utilities, or research ergonomics. Changes should
not turn Talibus into a production poker bot, live-play assistant, overlay,
real-money gambling tool, casino automation system, or platform-rule bypass
tool.

## Responsible-Use Rules

- Do not add features intended for live poker decision support.
- Do not add poker-site automation, scraping, account automation, or overlay
  workflows.
- Do not frame simulator results as real-money performance.
- Do not claim solved poker, guaranteed profitability, human-level strength,
  solver-level strength, or proven multiplayer Deep CFR convergence.
- Keep public claims tied to controlled simulator evaluation and documented
  limitations.

## Basic Checks

From the repository root:

```bash
python -m pip check
python -m compileall -q training/deep_cfr eval run_eval_suite.py
python -m unittest discover eval
python -m unittest discover -s training/deep_cfr -p "test_*.py"
python run_eval_suite.py --help
python -m eval.run_league --help
python training/deep_cfr/run_deep_cfr.py --help
python training/deep_cfr/run_smoke.py --help
```

From `solver/`:

```bash
cargo +stable fmt --all -- --check
cargo +stable clippy --workspace --all-targets --locked
cargo +stable test --workspace --locked
```

These commands match the source, CLI, and unit checks in the Python/Rust CI
workflows. For a real CPU integration check after installing both Python
requirement files, run from the repository root:

```bash
python training/deep_cfr/run_smoke.py
```

See [CPU Smoke Experiment](docs/smoke.md) for generated artifacts and
reproducibility limits. The full smoke is separate from normal CI because it
requires export/runtime dependencies and a Rust release build.

## Suggested Contribution Areas

- Documentation and reproducibility notes.
- Evaluation harness improvements.
- Rust engine tests.
- Python training utilities.
- Smoke-run and result-pack validation scripts.
- Experiment metadata and artifact tracking.
- Conservative limitation and responsible-use documentation.

## Pull Request Guidance

Keep changes narrow enough to review. Describe the problem, resulting behavior,
important design decisions, verification commands and outcomes, and remaining
limitations. Link relevant issues and evidence; distinguish checks actually run
from planned checks. If a change affects result interpretation, update
[Evaluation](docs/evaluation.md), [Limitations](docs/limitations.md), and
[Responsible Use](docs/responsible-use.md) as needed.

## Issues And Repository Maintenance

- Write issue titles around the engineering problem. Include technical context,
  scope, acceptance criteria, and dependencies where relevant.
- Use descriptive branches such as `fix/seed-initialization`,
  `docs/runtime-setup`, or `chore/repository-hygiene`. Prefer technical commit
  subjects with prefixes such as `fix:`, `test:`, `docs:`, or `ci:`.
- Keep public descriptions focused on technical decisions and evidence. Keep
  private planning notes, assistant prompts, and personal learning notes outside
  the repository. Preserve attribution and development provenance, including
  relevant AI-assistance disclosures; do not imply reviews or checks occurred
  without evidence.
- Update stale status instructions after completion while retaining links,
  measurements, limitations, and substantive discussion.
- Delete a merged branch only after checking for work added after its PR merged
  and confirming that its changes are retained on the default branch. Preserve
  active/protected branches, published history, tags, and releases.
