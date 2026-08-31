# Acceleration integration contract

GPU acceleration is a later implementation phase. CI is established first so
the accelerated backend has an explicit compatibility target.

## User experience

The existing call remains valid:

```python
mde = MDE( data, target = "FWD", D = 5 )
mde.Run()
```

The preferred final design is automatic backend selection with an optional
explicit override for reproducibility. CPU-only installation and execution
must continue to work without importing a GPU framework. Accelerator
dependencies belong in an optional package extra and are imported lazily.

The exact configuration name will be chosen after reviewing the provided GPU
implementation; CI should not freeze a speculative API before that review.

## Correctness gates

An accelerated implementation is mergeable only after it demonstrates:

- the same public `MDE`, `MDEConfig`, CLI, and `MDEOut` schema;
- identical selected-variable order for deterministic fixtures;
- numerical agreement at a documented tolerance for rho, embedding dimension,
  CCM qualification, and precomputed slope-matrix paths;
- fixed-seed repeatability;
- CPU fallback when no supported device or optional dependency is present;
- clear failure for an explicitly requested but unavailable device;
- parity for `noCCM`, fixed and automatic `E`, `Tp`, `tau`, exclusion radius,
  missing values, and constant inputs.

## Performance gates

Correctness is required on ordinary hosted CI. GPU benchmarks belong in a
manual workflow on a labelled self-hosted runner. Initially they report timing,
device, data shape, dependency versions, warm-up, and speedup as artifacts;
they do not fail on a speed threshold until variance on stable hardware is
known.

The historical `fastccm` branch may inform adapter design, but it is not a merge
base: it predates current `MDEConfig`, parallel execution, and `ReverseMDE`.
