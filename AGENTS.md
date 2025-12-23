# Repository Guidelines

## Project Structure & Module Organization
- Core library lives in `src/openpi`: `models` (JAX), `models_pytorch`, `policies`, `training`, `serving`, shared utils in `shared`.
- User-facing scripts: `scripts/train.py`, `train_pytorch.py`, `compute_norm_stats.py`, `serve_policy.py`.
- Workflows and robot docs: `examples/`; move long explanations to `docs/`.
- Workspace dependencies: `packages/` (uv workspace). Add new Python subpackages under `packages/<name>/` with a `pyproject.toml`.

## Build, Test, and Development Commands
- Install deps: `GIT_LFS_SKIP_SMUDGE=1 uv sync && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .` (needed for LeRobot submodule).
- Lint/format: `uv run ruff check .` and `uv run ruff format .` (line length 120).
- Pre-commit: `uv run pre-commit run --all-files`.
- Tests: `uv run pytest` (add `-k <pattern>`). Keep fast CPU-friendly cases.
- Training (JAX): `uv run scripts/train.py <config_name> --exp-name=<run>`; compute stats first with `uv run scripts/compute_norm_stats.py --config-name <config_name>`.
- Training (PyTorch): `uv run scripts/train_pytorch.py <config_name>`.
- Serve a policy: `uv run scripts/serve_policy.py policy:checkpoint --policy.config=<name> --policy.dir=<ckpt_dir>`.

## Coding Style & Naming Conventions
- Python 3.11; use 4-space indent, type hints, and prefer explicit imports.
- Follow `ruff` rules (see `pyproject.toml`); keep functions short and log via the existing logging utilities.
- Naming: snake_case for modules/functions, CapWords for classes, UPPER_SNAKE for constants; keep config names aligned with files in `training/config.py`.
- Place sizeable utilities in `shared/`; keep robot- or model-specific logic inside their respective subpackages.

## Testing Guidelines
- Use `pytest`; co-locate tests under a nearby `tests/` folder mirroring the source path (create it if absent).
- Provide deterministic seeds for JAX/torch where applicable; prefer small fixtures and synthetic data to avoid large downloads.
- When adding configs, include a minimal smoke test that exercises the new path through `scripts/train.py --dry-run` or a short `pytest` that builds the config objects.

## Commit & Pull Request Guidelines
- Commit messages: concise, imperative, optionally scoped (e.g., `chore: fix typos`, `train: add libero config`). Avoid long bodies unless needed.
- Before opening a PR: ensure `ruff check`, `ruff format`, and `pytest` pass; note GPU/CPU requirements for any new code.
- PR description should state motivation, summarize changes, list breaking behavior, and link issues/discussions. Add logs or screenshots for UI/visual changes and sample training output for pipeline changes.
- Avoid committing large artifacts or checkpoints; place data under a cache path (`~/.cache/openpi`) and document any new env vars (e.g., `OPENPI_DATA_HOME`).
