# Repository instructions

These instructions apply to the entire repository.

## Instruction hierarchy

- Nested `AGENTS.md` files supplement this file for their directory trees. Follow every applicable
  file from the repository root to the file being changed; the nearest file governs only when a
  local rule is more specific.
- Keep nested files limited to durable, directory-specific guidance. Do not repeat repository-wide
  rules or use them for temporary task status.
- Task-level `AGENTS.md` files are tracked documentation. Keep personalized task artifacts under
  the task's `files/` or `outputs/` directory and name them `*.local.*` so they remain untracked.

## Working conventions

- Use Python 3.12 and `uv`; prefer `uv run` over invoking `python3` directly.
- If managed macOS sandboxing prevents `uv` from reading its global cache, use the existing
  environment without syncing:
  `UV_CACHE_DIR=.uv-cache UV_OFFLINE=1 uv run --no-sync <command>`.
- Search with `rg` or `rg --files`.
- Preserve unrelated workspace changes. Inspect the current worktree and remotes rather than
  assuming a clean checkout or a configured upstream.
- Treat `main` as the authoritative deliverable. Agents may use temporary worktrees or feature
  branches for isolation and may merge or cherry-pick completed work into `main` without additional
  approval; do not leave required work only on a side branch.
- Keep the core package offline. Network access belongs only in explicit adapters such as
  `YFinancePriceProvider`.
- Matplotlib, seaborn, and yfinance are available through optional extras when the task needs them.
  Prefer Tectonic for LaTeX work; verify availability before attempting to install another engine.

## Source of truth

- Read `ARCHITECTURE.md` for module ownership and extension boundaries.
- Read `MODEL_NOTES.md` before changing financial formulas, compatibility behavior, calibration,
  tax assumptions, or stochastic-model semantics.
- `README.md` owns user-facing installation and API guidance.
- `pyproject.toml` owns supported Python, dependencies, lint, typing, test, and coverage settings.
- `src/lifecycle_finance/calibrated_regime_defaults.py` is generated. Regenerate it with
  `tools/calibrate_regime_model.py`; do not edit fitted constants by hand.

## Behavioral invariants

- `RegimeSwitchingMarket` is the default return model. `StochasticMarket` remains the legacy
  compatibility model.
- Seeded runs are deterministic for a fixed configuration. Chunk size is part of a chunked run's
  deterministic configuration.
- Keep market generation vectorized over paths. Do not add per-path Python loops or retain
  path-by-horizon shock or regime histories.
- Antithetic pairs share discrete regime and epistemic state and use opposite continuous shocks.
  Transition only when another modeled year remains.
- Unlevered asset returns respect limited liability, and configured real-rate floors apply from
  the initial state onward.
- `with_config()` replaces a complete market configuration.
  `with_config_overrides()` changes named fields while preserving model-specific calibration.
- User preferences are finite utility curves, not feasibility constraints. Insolvency, policy
  shortfall, and preference breaches remain separate diagnostics.
- Aggregate Monte Carlo wealth and tax-lot account projections are separate engines; do not
  silently mix their accounting.
- `WorkbookSocialInsurance` preserves workbook retirement-age behavior.
  `SocialSecurityPolicy` is the modern claim-age-aware implementation.
- Financial inputs must reject nonfinite values rather than allowing NaN or infinity to propagate.
- Do not change public defaults, statistical acceptance thresholds, or calibrated constants merely
  to make a test pass.

## Changes and tests

- For bugs, reproduce the failure first and add a focused regression test before production code.
- Test observable behavior with real components. Avoid tests that only mirror implementation
  details or assert on mocks.
- Preserve public exports in `src/lifecycle_finance/__init__.py` and documented compatibility
  imports unless the task explicitly changes the API.
- Run focused tests while iterating. Before handing off code, run:

```bash
uv run ruff check src tests
uv run mypy src
uv run pytest -q
```

- Use the sandbox-safe `uv` form above when normal `uv run` is unavailable.
- Run `uv build` when changing packaging, exports, dependencies, or release artifacts.
- Coverage must remain at or above the threshold in `pyproject.toml`.

## Git and prose

- Stage selectively and keep commits scoped. Do not add AI-generation branding to commits or pull
  request descriptions.
- Do not commit caches, local task inputs or outputs, generated plots, build artifacts, or secrets.
- Human-facing prose should be terse and reader-ordered. Document non-obvious rationale and
  cross-cutting invariants; point to code, Git, or another source of truth for facts they own.
- Remove completed-work narratives and stale status snapshots instead of preserving them as
  historical documentation.
