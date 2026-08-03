# Architecture

The package separates assumptions, deterministic planning, stochastic paths, and implementation
adapters so each can be tested independently.

| Module | Responsibility |
|---|---|
| `domain` | Validated immutable inputs and result records |
| `demographics` | Gompertz survival, death distributions, and annuity factors |
| `income` | Salary curves, income projection, and Social Security |
| `income_risk` | Income-state transitions for stochastic path generation |
| `taxes` | Progressive ordinary/LTCG tax and annual tax ledgers |
| `markets` | Capital-market assumptions and covariance validation |
| `return_models` | Market-path protocol plus legacy and regime-switching return implementations |
| `regime_calibration` | Offline historical fit and epistemic bootstrap generation |
| `utility` | Bounded curves, named add-ons, decomposition, and composite decision utility |
| `calibration` | Pairwise indifference, equivalent spending, and marginal trade-offs |
| `decisions` | Joint rolling spending, insurance, allocation, and leverage optimization |
| `lifecycle` | Human capital, liabilities, consumption, insurance, and balance sheet |
| `allocation` | Human-capital-aware unconstrained/constrained financial allocations |
| `spending` | Analytic marginal-utility solver and comparison policies |
| `accounts` | Tax lots, accounts, contributions, conversions, withdrawals, and rebalancing |
| `simulation` | Vectorized Monte Carlo engine |
| `policy_paths` | Reusable vectorized evaluation of supplied market paths and policy decisions |
| `sweeps` | Common-random-number parameter sweeps |
| `serialization` | Stable JSON-compatible input and result conversion |
| `cli` | Thin command-line adapter |

Extension points use `typing.Protocol`: utility curves, income models, spending policies, return
models, and allocation constraints can be replaced without subclassing internals.

`simulation` owns lifecycle path accounting and consumes the `return_models` protocol.
`RegimeSwitchingMarket` is the default implementation; `StochasticMarket` remains the legacy
compatibility implementation. Each model owns distinct default assumptions. `with_config()` is a
full replacement for sweeps that already hold the model's configuration;
`with_config_overrides()` changes named fields while preserving model-specific calibrated values.

The deterministic lifecycle result can seed the stochastic engine: its constrained allocation,
income path, survival model, consumption target, and bequest target are all first-class outputs.

## Decision invariant

A user preference is a `UtilityAddon(curve, typed outcome, importance, aggregation)`.
`UtilityModel.validate_outcomes()` compiles a complete decision problem and rejects unavailable
outcomes. Narrow internal subproblems select only their relevant components.

The deterministic planner uses composite utility when choosing its initial bequest. The stochastic
engine then reconsiders spending, incremental insurance, equity exposure, and leverage as state
changes. Sweeps may report any diagnostic but rank candidates only by mean total utility.

Physical insolvency, policy-funding shortfall, and preference breaches are independent result
types. A utility threshold never creates insolvency by itself.

Built-in curves provide derivatives and breakpoints to a piecewise root solver. Custom curves
fall back to bounded candidate search. Chunked summary simulation retains scalar path outcomes
while bounding annual-path and optimizer temporary memory.

Only implementability constraints remain hard. Long-only and account holding bounds describe
what can be executed; a desired allocation cap belongs in a utility curve instead.
