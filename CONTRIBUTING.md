# Contributing

MDE favors focused changes that preserve the public `dimx` API and the
scientific meaning of its reference outputs. Keep the existing naming and
layout conventions; avoid repository-wide formatting changes.

## Development setup

Use Python 3.11 or newer in an isolated environment:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --editable ".[test]"
```

Run fast tests while developing, then run the scientific regressions before
opening a pull request:

```bash
python -m pytest -q \
  tests/test_Config.py tests/test_MDE_Unit.py \
  tests/test_ReverseMDE.py tests/test_CLI_Parser.py \
  tests/test_pyEDM_Compatibility.py \
  tests/test_ExternalValidationAdapter.py

python -m pytest -q tests/test_MDE.py tests/test_MDE_CCM_Matrix.py
```

See [Testing](docs/TESTING.md) and [Continuous integration](docs/CI.md) for
the complete test tiers and reproducibility settings.

## Pull requests

- Add or update a focused test for behavior changes.
- Keep stochastic scientific tests seeded whenever the API permits it.
- Do not regenerate golden outputs without documenting the numerical reason.
- Preserve CPU behavior when adding an optional accelerated backend.
- Include user-facing documentation for public API or dependency changes.
