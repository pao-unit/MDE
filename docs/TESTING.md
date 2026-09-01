# Testing

MDE uses layered tests so quick API failures are separated from longer
scientific validation.

## Test ownership

| Suite | Owner/source | Unique cases | Role |
| --- | --- | ---: | --- |
| Bundled scientific regressions | Pre-existing `pao-unit/MDE` tests | 5 | Authoritative MDE numerical behavior |
| Fast project tests | Added with this CI foundation | 37 | Configuration, API, CLI, orchestration, dependency, and adapter behavior |
| Independent validation | Pinned `pao-unit/EDM_MDE_validation` | 33 | External pyEDM and MDE conformance |

The five bundled tests retain upstream ownership. Three MDE golden tests were
strengthened to compare selected-variable identity and order, and the existing
CCM-matrix test now compares the three dimensions stated in its original
comment. The external test source and golden files are not modified.

## Reproducible environment

Install the project and its test extra in an isolated Python 3.11+ environment:

```bash
python -m pip install --editable ".[test]"
```

For deterministic, headless execution use:

```bash
export MPLBACKEND=Agg
export MPLCONFIGDIR="$PWD/.ci/matplotlib"
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONHASHSEED=0
mkdir -p "$MPLCONFIGDIR"
```

The native-thread limits prevent oversubscription when MDE or pyEDM starts
multiple Python processes. They are performance controls, not numerical
tolerances.

## Fast standard tests

```bash
python -m pytest -q \
  tests/test_Config.py \
  tests/test_MDE_Unit.py \
  tests/test_ReverseMDE.py \
  tests/test_CLI_Parser.py \
  tests/test_pyEDM_Compatibility.py \
  tests/test_ExternalValidationAdapter.py \
  --timeout=120 --durations=20
```

These tests should finish in seconds to roughly one minute and are run on the
minimum supported Python and the newest supported Python.

## Bundled scientific regressions

```bash
python -m pytest -q \
  tests/test_MDE.py tests/test_MDE_CCM_Matrix.py \
  --timeout=1800 --durations=20
```

All five repository-owned tests run on pushes and pull requests. Do not loosen
their rounding or equality rules simply to accommodate a platform difference;
first identify whether the difference is deterministic and scientifically
acceptable.

## Independent validation

The scheduled workflow checks out the pinned validation repository rather than
copying it. To reproduce it locally:

```bash
git clone https://github.com/pao-unit/EDM_MDE_validation.git external-validation
git -C external-validation checkout bae270e568dd52830f57ab661380e700097aa58d

python -m pytest -q -p ci.external_validation_adapter \
  external-validation/test_Simplex.py \
  external-validation/test_SMap.py \
  external-validation/test_CCM.py \
  external-validation/test_EDim.py

python -m pytest -q -p ci.external_validation_adapter \
  external-validation/test_MDE.py
```

The first command runs all 31 pyEDM cases without editing their source. The
adapter restores pyEDM's shared Lorenz sample after every case because the
external `test_simplex7` modifies it in place; with test isolation all 31 pass.
The second command runs both MDE cases with the documented legacy-keyword
translation; its historical Lorenz case is a strict expected failure and its
Fly case passes. See [test provenance](UPSTREAM_BASELINE.md).

## Packaging

```bash
python -m pip install build twine
python -m build
python -m twine check dist/*
```

CI additionally installs the built wheel outside the source checkout and
checks `dimx.__version__` plus both packaged example data files.
