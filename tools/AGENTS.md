# Tooling instructions

These rules supplement the repository instructions for `tools/`.

- Tools are explicit maintenance entry points, not hidden import-time behavior. Give new tools a
  narrow CLI, deterministic defaults, useful validation errors, and a nonzero exit on failure.
- `calibrate_regime_model.py` owns regeneration of the fitted regime defaults. Change the fitting
  method or source handling in the tool, regenerate the module, and review both diffs together.
- Keep calibration reproducible and offline by default. Record input provenance and configuration
  in generated source comments rather than relying on an analyst's local environment.
- Never overwrite a generated target until all source data and fitted values have passed validation.
- Do not commit scratch datasets, downloaded market data, caches, or one-off tool output.
