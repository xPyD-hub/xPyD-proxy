# Contributing to MicroPDProxy

Thank you for your interest in contributing! This guide covers the workflow and
standards we follow.

## Branch Strategy

1. Fork or create a **feature branch** from `main`:
   ```bash
   git checkout -b feature/my-change main
   ```
2. Make your changes, commit with clear messages.
3. Open a **Pull Request** against `main`.
4. All PRs require at least one approving review before merge.

## Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting, and
[isort](https://pycqa.github.io/isort/) for import sorting.

```bash
# Check
ruff check .
isort --check-only .

# Auto-fix
ruff check --fix .
isort .
```

## Testing

- All CI checks **must pass** before a PR can be merged.
- Run the test suite locally:
  ```bash
  pip install -r requirements.txt
  python -m pytest tests/ -v
  ```
- If you add a new feature, add corresponding tests.

## Review Process

1. Open a PR with a clear description of **what** and **why**.
2. At least one maintainer must approve.
3. CI must be fully green.
4. After approval, the PR will be merged (squash or merge commit).

## Changes to `core/`

The `core/` directory contains the validated proxy implementation. Changes here
are held to a **higher standard**:

- Must not break existing topology matrix tests
  (`tests/test_proxy_matrix.py`).
- Must include test coverage for new behavior.
- Require review from a core maintainer.
- Performance-sensitive changes should include benchmark results.

## Reporting Issues

Open a GitHub issue with:
- Steps to reproduce
- Expected vs. actual behavior
- Environment details (Python version, OS, GPU setup if relevant)
