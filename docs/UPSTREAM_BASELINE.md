# Upstream baseline and test provenance

The CI foundation was created from a clean clone of
[`pao-unit/MDE`](https://github.com/pao-unit/MDE) rather than from an existing
working directory.

| Source | Branch | Pinned revision | Role |
| --- | --- | --- | --- |
| `pao-unit/MDE` | `main` | `77ca953799b6790403a052d4ce1a19c61e118a6c` | Source and bundled tests |
| `pao-unit/EDM_MDE_validation` | `main` | `bae270e568dd52830f57ab661380e700097aa58d` | Independent numerical validation |

The baseline was recorded on 2026-08-27. Updating either pin requires a review
of changed tests, golden outputs, dependencies, and runtime before the workflow
is updated.

## Bundled MDE tests

The five pre-existing tests remain in their original files and all run in the
required scientific-regression CI job:

1. Lorenz5D MDE golden output.
2. Fly80 forward-motion MDE golden output.
3. Fly80 left/right-motion MDE golden output.
4. Fly80 `Evaluate` prediction golden output.
5. Internal CCM versus a precomputed CCM slope matrix.

These are integration-level numerical regressions. The added fast tests cover
configuration, validation failures, CLIs, reverse traversal and output, and the
pyEDM call interface without replacing the scientific suite. The three MDE
golden tests now assert selected-variable order as well as rho, and the slope
matrix test compares the three dimensions its original comment specified.

The v1.2.0 multiprocessing refactor (`69c59ae`) introduced a second leading-
column removal after `removeTime=True` had already removed the time vector. On
Lorenz5D this silently removed `V1`, and the same commit deleted the fourth row
from the bundled golden. The pre-refactor bundled golden and independent
validation golden both require `V1,0.976646`. This fork makes numeric-frame
preparation the sole owner of leading-column removal, leaves the input frame
and the `removeTime` / `noTime` settings unchanged, restores that fourth row,
and covers all four flag combinations in fast tests.

## Independent validation suite

`EDM_MDE_validation` contains 33 pytest cases and 31 golden-output files:

- 14 `Simplex` tests;
- 4 `SMap` tests;
- 6 `CCM` tests;
- 7 `EmbedDimension` tests;
- 2 MDE tests.

The validation repository has no license file, so its source is not vendored
here. CI checks out the pinned commit and runs every test. Its two MDE tests
predate `MDEConfig` and still pass `cores` and `title`. The small pytest hook in
`ci/external_validation_adapter.py` translates `cores` to `crossMapCores` and
removes the unused plotting title at runtime; the external tests and reference
outputs remain unchanged.

The external `test_simplex7` inserts NaNs directly into the shared
`pyEDM.sampleData["Lorenz5D"]` frame. If all files run in one process without
isolation, those NaNs change `test_smap4`, `test_ccm5`, and EmbedDimension cases
1, 6, and 7. Each of those five cases passes against its golden when started
with pristine data. The pytest hook snapshots and restores the Lorenz sample
after every case, so all 31 external pyEDM tests pass without editing their
source or reference outputs.

Both external MDE cases pass with only the legacy-keyword adapter. The Lorenz
case is a regression oracle for the four-dimensional result restored above;
its source and golden remain unmodified.

Known source defects remain visible rather than silently rewriting an external
snapshot: `test_ccm3` builds NaN data but calls CCM with the clean frame,
`test_simplex7` mutates shared sample data, both MDE tests discard the
selected-variable column, and neither MDE test fixes its CCM seed. The hook
contains the shared-data side effect; project-owned tests cover behavior that
cannot be trusted to the remaining cases.
