# Core package instructions

These rules supplement the repository instructions for `src/lifecycle_finance/`.

## Boundaries

- Put behavior in the module that owns it according to `ARCHITECTURE.md`; prefer a small protocol
  or adapter over making one engine understand another engine's accounting.
- Keep the package offline. Accept externally acquired data only through explicit provider
  boundaries and validated domain records.
- Preserve aggregate Monte Carlo, policy-path, and tax-lot account engines as separate accounting
  systems. Shared inputs do not imply shared balance-update logic.

## APIs and data

- Use immutable validated records for financial inputs and public results. Copy caller-owned NumPy
  arrays when retaining them and reject wrong rank, wrong shape, nonfinite values, and invalid
  probability or fraction ranges at the boundary.
- Keep path and candidate work vectorized. A loop over modeled years or a small fixed set of models
  is acceptable; a Python loop over simulation paths or optimization candidates is not.
- Preserve seeded draw cadence and deterministic configuration when changing stochastic code.
- Update package-root imports and `__all__` for intended public APIs. Preserve documented
  compatibility imports unless the task explicitly removes them.
- Do not edit `calibrated_regime_defaults.py`; regenerate it with the calibration tool.

## Verification

- Add focused tests in the corresponding `tests/test_<area>.py` file. Protocol changes need both
  direct boundary tests and a real-consumer integration test.
- Changes to stochastic formulas need hand-checkable deterministic cases before statistical tests.
- Run `mypy` on changed package modules while iterating; do not defer type cleanup to the full gate.
