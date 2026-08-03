# Benchmark instructions

These rules supplement the repository instructions for `benchmarks/`.

- Benchmarks measure performance; tests establish correctness. Keep benchmark assertions limited to
  configuration and result sanity, and retain correctness coverage in `tests/`.
- Compare implementations with identical seeds, path counts, horizons, chunk sizes, and retained
  outputs. Warm up both sides consistently before timing.
- Report enough configuration for another run to reproduce the result. Do not publish a speedup
  without the baseline time and workload.
- Do not improve numbers by weakening the workload, skipping validation only on one side, or
  changing statistical behavior.
- Do not commit generated profiles, timing dumps, plots, or machine-specific results.
