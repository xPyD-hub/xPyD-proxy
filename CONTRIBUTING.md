# Contributing to xPyD-proxy

## Development Setup

```bash
git clone https://github.com/xPyD-hub/xPyD-proxy
cd xPyD-proxy
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/unit/ -q
```

## Code Style

- Python 3.10+
- Ruff: `ruff check .`
- Pre-commit: `pre-commit run --all-files`
- All PRs must pass CI (lint + tests + integration trigger)

## Bot Development

See [bot/](bot/) for automated development policies.
