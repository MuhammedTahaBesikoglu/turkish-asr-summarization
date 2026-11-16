# Development workflow

## Lint and format

[ruff](https://docs.astral.sh/ruff/) (pinned in `requirements-dev.txt`) handles both. Rules are
configured in `pyproject.toml` (`E`, `W`, `F`, `I`, `UP`, `B`, `SIM`; line length 100; target
Python 3.12). Notebooks are included.

```bash
ruff check .          # lint
ruff format .         # format
ruff format --check . # what CI runs
```

## Tests

```bash
pytest
```

- `tests/test_smoke.py` checks that `src` and every subpackage import cleanly.
- Tests must not require PyTorch, a GPU, model downloads or the dataset, so they run in CI.
  Anything that needs those belongs in a script, not in the test suite.

## CI

`.github/workflows/ci.yml` runs on every push to `main` and on pull requests, on Python 3.12:

1. Install `requirements-dev.txt` and the package (`pip install -e .`), not the ML stack.
2. `ruff check .`
3. `ruff format --check .`
4. `pytest`

## Conventions

- Small, meaningful commits; every stage ends in a working state.
- Hyperparameters and paths live in `configs/*.yaml`.
- Core logic goes in `src/`; notebooks are for analysis only.
- Each component gets a doc in `docs/` (see [README.md](README.md)).
