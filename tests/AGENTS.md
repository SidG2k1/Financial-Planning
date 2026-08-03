# Test-suite instructions

These rules supplement the repository instructions for `tests/`.

- Test observable financial behavior and public contracts. Use internal implementation details only
  when the unit itself is private and no public observation can isolate the behavior.
- For bugs, keep the smallest reproduction that fails before the fix. Prefer literal,
  hand-calculable fixtures over fixtures derived with the production formula.
- Use fixed seeds and exact array assertions for deterministic behavior. Statistical tests must
  explain the property being measured and retain the acceptance thresholds owned by the model.
- Exercise real planners, markets, policies, and engines where integration matters; use small test
  doubles only to probe a protocol boundary or force a specific edge case.
- Reject warnings and noisy output. Do not make a test pass by loosening tolerances, reducing path
  counts, or changing public defaults without a model-level justification.
- Keep network access out of the suite. Price-provider tests should use supplied local data or a
  deterministic provider implementation.
- Tests for ignored `tasks/*/files/*.local.*` scripts stay beside those scripts and are run
  explicitly with `--import-mode=importlib`; they do not belong in the default package test
  collection.
