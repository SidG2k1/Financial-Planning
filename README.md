# lifecycle-finance

One composable Python library for:

- lifecycle balance sheets: financial wealth + human capital − consumption liabilities;
- survival-conditioned consumption, annuitization, insurance, and bequest planning;
- human-capital-aware financial glide paths;
- stochastic retirement simulations with fat tails, stochastic volatility, income shocks,
  uncertain lifespan, and several leverage instruments;
- US tax, account, lot, Roth-conversion, withdrawal, and asset-location analysis;
- reproducible Monte Carlo sweeps and optimization.

All monetary values are real annual dollars. Rates are decimals. The core package is offline and
deterministic when given a seed. Network market data and plotting are optional adapters.

## One objective: composable utility

Preferences are utility curves, not hard decision rules. For example, “I want at least $40,000
of annual spending” is a finite penalty below $40,000:

```python
from lifecycle_finance import (
    OutcomeType,
    SpendingFloorCurve,
    UtilityAddon,
    UtilityAggregation,
)

spending_need = UtilityAddon(
    name="basic_spending",
    outcome=OutcomeType.SPENDING,
    curve=SpendingFloorCurve(threshold=40_000, scale=10_000),
    importance=2.0,
    aggregation=UtilityAggregation.DISCOUNTED_MEAN,
)
```

The unweighted curve is 0 at and above $40,000, -1 at $30,000, and -4 at $20,000.
`importance` states how much that shape matters relative to consumption, bequest, retirement
timing, leverage, and other add-ons. Curves remain finite so every preference is tradeable and
inspectable; `plot_utility_curve(spending_need, values)` renders the weighted curve. Hard
constraints are reserved for feasibility: accounting identities, unavailable account holdings,
and limited liability.

`PlanningScenario().preferences` includes this $40,000 curve by default. Supply additional
`UtilityAddon` objects to `LifecyclePlanner` or `MonteCarloEngine`, or put serializable
`UtilityAddonConfig` records on the scenario. Outcomes are typed and validated before a complete
simulation is scored. Time-varying add-ons explicitly select discounted mean, discounted sum,
worst-year, or last-observation aggregation.

Retirement freedom and work quality are annual flows rather than target ages:

```python
from lifecycle_finance import LinearCurve

retirement_freedom = UtilityAddon(
    name="retirement_freedom",
    outcome=OutcomeType.RETIRED,
    curve=LinearCurve(),
    importance=0.53,
    aggregation=UtilityAggregation.DISCOUNTED_SUM,
    age_reference=40,
    age_growth=0.02,
)
```

Simulations expose aligned `RETIRED` and `WORKING` path-year outcomes. Unless overridden,
allocation risk tolerance is derived from the consumption elasticity so rolling allocation and
CRRA consumption utility use the same risk aversion.

Simulations separate physical insolvency, inability to fund the selected policy, and preference
breaches. Sweeps optimize total utility even when displaying one of those diagnostics.

## Install and run

This project is initialized as a local Git repository. Configure a remote separately if you want
to publish it.

```bash
uv sync --extra dev
uv run lifecycle-finance example
uv run lifecycle-finance plan
uv run lifecycle-finance simulate --paths 1000 --seed 42
uv run lifecycle-finance simulate --paths 100000 --chunk-size 10000 --summary-only
uv run pytest
```

## Public API

```python
from lifecycle_finance import (
    FinancialWealth,
    LifecyclePlanner,
    Person,
    PlanningScenario,
    Sex,
)

scenario = PlanningScenario(
    person=Person(current_age=45, retirement_age=66, sex=Sex.FEMALE),
    wealth=FinancialWealth(
        domestic_equity=200_000,
        global_equity=250_000,
        bonds=600_000,
        cash=150_000,
    ),
)
plan = LifecyclePlanner().plan(scenario)
print(plan.net_worth, plan.constrained_allocation)
```

The package uses small immutable domain objects, explicit assumption objects, protocols for
extension points, and plain NumPy arrays at numerical boundaries. JSON serialization is included
for application and notebook workflows.

### Supplied market paths and policies

`MarketPaths` return arrays are path × return-year. A `PolicyPathPolicy` sees beginning-of-year
state, not the current year's returns; result histories are candidate × path × modeled-year.
Each year applies release, income and vesting, ordinary spending, event spending, allocation,
recording, then returns. `target_total_equity` includes restricted equity, while
`liquid_equity_weight` describes only the liquid account.

```python
import numpy as np

from lifecycle_finance import (
    MarketPaths,
    PlanningScenario,
    PolicyPathContext,
    PolicyPathDecision,
    PolicyPathEvaluator,
    UtilityModel,
)


class FixedPolicy:
    def decide(self, context: PolicyPathContext) -> PolicyPathDecision:
        return PolicyPathDecision(
            external_income=50_000,
            consumption=40_000,
            target_total_equity=0.60,
        )


zeros = np.zeros((2, 2))  # two paths × two return years
market = MarketPaths(
    equity_returns=zeros,
    bond_returns=zeros.copy(),
    cash_returns=zeros.copy(),
    real_rates=zeros.copy(),
)
result = PolicyPathEvaluator(
    UtilityModel.from_scenario(PlanningScenario())
).evaluate(
    market,
    FixedPolicy(),
    candidate_count=1,
    ages=(45.0, 46.0, 47.0),
    exposure=(1.0, 1.0, 1.0),
    initial_liquid_wealth=500_000,
    initial_restricted_equity=100_000,
)
```

## Task workspaces

Keep objective-specific work under:

```text
tasks/<objective>/files/
tasks/<objective>/outputs/
```

`files/` holds task inputs, source-derived state, and runnable scenarios. `outputs/` holds
generated analyses, recommendations, reports, and exports. Use concise kebab-case objective
names. Personal or otherwise untracked files must include `.local.` in the filename; local files
under both directories are ignored. Core library code, tests, and general documentation stay
outside task workspaces.

Local analytical outputs are reproducible only with the exact task script, inputs, and cached
random worlds that produced them. They are working artifacts, not package verification fixtures.

### Monte Carlo

```python
from lifecycle_finance import (
    LeverageInstrument,
    MonteCarloEngine,
    SimulationSettings,
)

result = MonteCarloEngine().simulate(
    scenario,
    settings=SimulationSettings(
        paths=10_000,
        seed=42,
        leverage=1.5,
        leverage_instrument=LeverageInstrument.BOX_SPREAD,
    ),
)
print(result.summary())
```

The path engine is vectorized. The default rolling optimizer chooses path-specific spending and
incremental insurance under one utility model. Equity exposure and leverage are one cohort-level
choice per simulated year, applied to every active path in that simulation call or chunk; they
are not per-path feedback controls. Insurance premiums consume cash when purchased and benefits
arrive at simulated death. `SimulationSettings.leverage` is the maximum available leverage, not
a forced target; the optimizer considers that exact cap and the selected instrument’s modeled
costs. Terminal bequest and insured bequest are separate outcome types.

### Income-risk models

`MonteCarloEngine` accepts an `IncomeRiskModel`. Its default is the one-year,
market-correlated `TransitoryMarketJobLoss`; supply a model explicitly when an analysis needs
persistent income effects:

```python
import numpy as np

from lifecycle_finance import (
    MonteCarloEngine,
    PersistentDisplacementIncomeRisk,
    generate_income_risk_paths,
)

income_risk = PersistentDisplacementIncomeRisk(
    baseline_probability=0.02,
    market_sensitivity=4.0,
    probability_cap=0.25,
    income_fractions_after_displacement=(0.50, 0.75, 1.00),
)
engine = MonteCarloEngine(
    income_risk_model=income_risk,
)
paths = generate_income_risk_paths(
    income_risk,
    deterministic_income=np.full(3, 50_000.0),
    equity_returns=np.zeros((1_000, 3)),
    real_rates=np.zeros((1_000, 3)),
    equity_risk_premium=0.04,
    working_years=3,
    random_uniforms=np.full((1_000, 3), 0.5),
)
assert paths.realized_income.shape == (1_000, 3)
```

`realized_income`, `income_fraction`, `displaced`, and `vesting_eligible` are each path ×
modeled-year arrays.

The default return model is a three-state regime-switching multivariate Student-t model for
equity, bonds, and cash. Its normal, growth/deflation-stress, and inflation-stress states change
expected returns, volatility, joint tails, and cross-asset covariance; the two stress states
produce opposite stock–bond correlation signs. Equity is generated in bounded log-return space,
so an unlevered annual equity return cannot fall below −100%. Persistent regimes, stochastic
volatility, and mean-reverting rates create serial dependence without directly autoregressing
returns.

Each Monte Carlo path selects one fitted hyperparameter scenario at inception and keeps it for
the full horizon. Those bootstrap scenarios represent epistemic uncertainty in quantities such
as the equity risk premium, conditional volatilities, transition probabilities, and initial
regime probabilities. Annual regime transitions and shocks remain aleatory uncertainty.
Antithetic path pairs share the same hyperparameter and regime paths.

`RegimeModelConfig` holds the fitted centers and uncertainty ensemble. The default regime model's
`MarketModelConfig` reflects that fit; changing its `equity_risk_premium` shifts every regime
premium, while equity and rate volatility overrides scale the fitted regime values. Transition
or initial-probability changes require a matching epistemic ensemble or `epistemic=None`.
Student-t tails and stock-bond correlations are regime-specific rather than global overrides.
`StochasticMarket` and a standalone `MarketModelConfig()` retain their legacy defaults. Select
that compatibility model with `--market-model legacy`; `--market-model regime` is the default.

```bash
uv run lifecycle-finance simulate --paths 8 --market-model regime
uv run lifecycle-finance simulate --paths 8 --market-model legacy
```

The checked-in regime defaults are fitted offline to annual US real returns through 2025.
Simulation performs no download or calibration. To reproduce the generated constants from the
source workbook:

```bash
uv sync --extra calibration
uv run python tools/calibrate_regime_model.py histretSP.xls --check
```

The generated module records the source URL, file digest, sample years, and deterministic
bootstrap seed. The fit is a historical model with substantial estimation uncertainty, not a
current-market forecast.

Built-in utility curves expose analytic marginal utility. Custom curves without derivatives use
the bounded grid fallback. `simulate_chunked()` supports full-path or scalar-outcome results;
summary mode retains exact path-level terminal wealth, certainty equivalents, utility, and
diagnostics. Exact built-in annual curves stream without annual result histories; custom annual
curves and subclasses use the full-history compatibility path. Seeds reproduce a fixed chunking
configuration; changing the chunk size intentionally creates a different deterministic set of
child streams. Summary-only online reductions are deterministic but may differ from full-history
results at floating-point rounding scale. Market returns and real rates remain
chunk-by-horizon matrices, so temporary memory still scales with `chunk_size × horizon`.

### Calibrate preference importance

```python
from lifecycle_finance import UtilityCalibrator, UtilityModel

model = UtilityModel.from_scenario(scenario, [spending_need])
calibrator = UtilityCalibrator(model)
calibration = calibrator.calibrate_importance(
    "basic_spending",
    option_a,
    option_b,
)
calibrated_model = calibrator.with_calibrated_importance(calibration)
```

`option_a` and `option_b` are `UtilityOutcome` records the user considers equally desirable.
Calibration solves the add-on importance that makes the statement true. The same API reports
equivalent constant annual spending and marginal rates of substitution.

### Accounts and asset location

```python
from lifecycle_finance import (
    Account,
    AccountPortfolio,
    AccountType,
    Asset,
    TaxLot,
    rebalance_asset_location,
)

assets = {
    "STOCK": Asset("STOCK", "equity", expected_real_return=0.05),
    "BOND": Asset("BOND", "bonds", expected_real_return=0.02),
}
portfolio = AccountPortfolio(
    accounts={
        "401k": Account(
            "401k",
            AccountType.TRADITIONAL,
            [TaxLot("BOND", shares=1_000, basis=100_000, acquired_year=2020)],
            allowed_assets=frozenset(assets),
        )
    },
    prices={"STOCK": 100.0, "BOND": 100.0},
)
rebalance_asset_location(
    portfolio,
    assets,
    {"equity": 0.6, "bonds": 0.4},
    year=2026,
)
```

Account events mutate holdings and record modeled taxable sales, withdrawals, Roth conversions,
and investment income in explicit ledgers. Sales and rebalances require the model year;
withdrawals and Roth conversions return requested, executed, and shortfall amounts. A
contribution does not infer a tax deduction; callers must supply the intended cash flow and tax
treatment.

### Compatibility and current policy

`LifecyclePlanner` defaults to `WorkbookSocialInsurance` for source parity. Its benefit formula
follows the workbook’s retirement-age logic; it is not a claim-age policy model. For analyses
that vary claim age or apply current policy, pass the separate 2026 real-dollar
`SocialSecurityPolicy` explicitly:

```python
from lifecycle_finance import LifecyclePlanner, SocialSecurityPolicy

planner = LifecyclePlanner(social_security=SocialSecurityPolicy())
```

Fixed nondiscretionary consumption and bequest settings are general inputs. The bundled
[workbook scenario](examples/workbook_scenario.json) uses them for source parity; use utility
curves for decisions.

See [MODEL_NOTES.md](MODEL_NOTES.md) for deliberate differences from the source models.

## Design boundaries

- Monte Carlo uses aggregate financial wealth; account types, tax lots, and account events are a
  separate projection engine.
- The deterministic planner reproduces the workbook’s model structure, not its UI or macros.
- Account tax estimates are scenario analysis, not tax-return preparation.
- Projections are educational estimates, not investment advice.
- No default calculation silently downloads current data.

See [ARCHITECTURE.md](ARCHITECTURE.md) for module boundaries and extension points.

## Verification

```bash
uv run ruff check src tests
uv run mypy src
uv run pytest --cov=lifecycle_finance
uv build
```
