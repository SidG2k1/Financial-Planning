# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Monte Carlo retirement simulator that answers "When can I retire, and how much can I spend?" Compares three portfolio strategies (cash, stocks, leveraged stocks) over a configurable lifetime using lifecycle spending rules adapted to market conditions, vitality (health/energy decline with age), and Social Security.

## Running the Simulator

```bash
pip install -r requirements.txt

# Single simulation (year-by-year output)
python retirement_math.py --seed 42

# Monte Carlo (confidence intervals)
python retirement_math.py --monte-carlo 500

# Find optimal retirement age (main use case)
python retirement_math.py --retirement-sweep 500

# Optimal leverage analysis
python retirement_math.py --leverage-sweep 500
```

No formal test suite exists. Validate changes with:
```bash
python retirement_math.py --retirement-sweep 200
```

**Important:** In headless environments (CI, SSH, containers), set `MPLBACKEND=Agg` before running any script that imports matplotlib:
```bash
MPLBACKEND=Agg python retirement_math.py --retirement-sweep 200
MPLBACKEND=Agg python generate_advice.py
```

## Architecture

Modular flat-file layout with three composable layers. All files live in the project root (no package directory — `python retirement_math.py` works directly).

### File Structure

| File | Purpose | Lines |
|------|---------|-------|
| `retirement_math.py` | CLI entry point: `parse_args()` + `main()` | ~200 |
| `config.py` | `SimulationConfig`, `SimulationResult`, `post_tax`, `format_money` | ~170 |
| `models.py` | Vitality, SS benefit, Vasicek yields, Kelly, Gompertz mortality, Bayesian sampling | ~110 |
| `spending.py` | `YearContext`, `SpendingRule` ABC, `FixedSpending`, `AmortizedSpending`, `VitalityAmortizedSpending`, `MarginalUtilitySpending` | ~230 |
| `utility.py` | `UtilityScorer` ABC, `CRRAUtility` | ~80 |
| `simulator.py` | `run_simulation`, `_build_income_schedule`, `_generate_year_shocks` | ~200 |
| `sweeps.py` | `run_monte_carlo`, `run_leverage_sweep`, `run_retirement_sweep`, `run_2d_sweep`, CRN/antithetic helpers | ~400 |
| `plotting.py` | All `plot_*` functions (lazy matplotlib import) | ~200 |
| `portfolio.py` | `Portfolio` ABC, `CashPortfolio`, `StockPortfolio`, `LeveragedStockPortfolio`, `PortfolioManager` | ~150 |

### Dependency Graph (no circular imports)

```
config.py            (standalone)
  ↑
models.py            (imports config)
  ↑
spending.py          (imports config, models)
  ↑
utility.py           (imports config, models)
  ↑
portfolio.py         (standalone)
  ↑
simulator.py         (imports config, models, spending, portfolio)
  ↑
sweeps.py            (imports config, models, spending, utility, simulator)
  ↑
plotting.py          (imports config, models, sweeps)
  ↑
retirement_math.py   (imports all — thin CLI layer)
```

### Three Layers

1. **Financial Model** (`simulator.py:run_simulation`) — Stochastic market simulation: Vasicek mean-reverting bond yields, fat-tailed equity returns (Student-t, df=5), stochastic volatility (log-normal AR(1)), stock-bond correlation via discount rate channel, dynamic leverage borrowing costs.

2. **Decision Model** (`spending.py`) — Spending rules determining annual consumption:
   - `FixedSpending` — base expenses + lifestyle inflation
   - `AmortizedSpending` — lifecycle annuity over remaining life
   - `VitalityAmortizedSpending` — vitality-weighted annuity (legacy heuristic, c ∝ vitality)
   - `MarginalUtilitySpending` — **default**: Euler equation-optimal allocation that equalizes marginal utility per PV-dollar across ages. Uses `c_t ∝ (w_t / d^t)^γ` where `w_t = β^t · vitality · fire_mult` and `γ = 1/(1-α)` is the EIS. Spending floor reserves PV of minimum spending, then optimally allocates the excess.

3. **Utility Model** (`utility.py`) — CRRA power utility: `V = Σ β^t · vitality(age) · fire_mult · c^α`, scored via `CRRAUtility`. Certainty equivalent converts stochastic utility into constant annual spending.

## Key Conventions

- **Personal config via `.env`** — copy `.env.example` to `.env` for personal overrides (income, NW, expenses, etc.). `.env` is gitignored. CLI flags override `.env` values.
- **All monetary values are in $1,000 units (2023 USD)** — an income of `100` means $100k/year
- Ages are integers, rates are floats (0.05 = 5%)
- `SimulationConfig` (in `config.py`) is the central dataclass with 40+ parameters; new features should add fields here and wire through CLI flags in `parse_args()`
- Spending rules implement `SpendingRule.compute(ctx: YearContext) -> float`
- Utility scorers implement `UtilityScorer.score()` and `certainty_equivalent()`
- Portfolio classes implement `Portfolio.pass_year()` to simulate one year of returns
- Keep the three layers (Financial, Decision, Utility) decoupled

## Key Concepts

- **Vitality**: `v(age) = floor + (1-floor) * exp(-((age-peak)/half_life)^2)` — Gaussian health decay from peak (~30), floor 0.3
- **FIRE Multiplier**: Extra utility during retirement (default 1.8x) — higher values pull optimal retirement earlier
- **Spending Floor**: Reserves PV of minimum annual spending ($30k) before optimally allocating excess via marginal utility equalization
- **Retirement Tax Advantage**: 1.25x multiplier reflecting LTCG/Roth (~15%) vs earned income (~38%) tax rates
- **Kelly Optimal Leverage**: `L* = (ERP - spread) / σ²`
- **Margin Calls**: `LeveragedStockPortfolio` triggers forced liquidation when equity drops below `maintenance_margin` (default 25%), reducing leverage for a 2-year cooldown
- **Stochastic Lifespan**: Gompertz mortality `h(age) = a·exp(b·age)` — spending rules still plan for `expected_lifespan`, but simulation terminates at sampled death age (longevity risk)
- **Stochastic Income**: Market-correlated job loss `P(loss) = base · exp(sensitivity · max(0, -excess_return))` — income shocks during working years
- **Bayesian Parameter Uncertainty**: Per-simulation sampling of ERP, vol, bond yield from posteriors — captures estimation risk on top of market risk
- **Antithetic Variates**: Variance reduction by running each seed with both normal and negated shocks, averaging utilities
- **Common Random Numbers (CRN)**: Pre-generated seed lists reused across parameter values in all sweeps for smoother comparisons
- **2D Parameter Sweep**: Generic engine sweeping any two `SimulationConfig` fields with contour visualization

## Maintenance Rules

- **Always update CLAUDE.md and README.md when making feature changes** — keep documentation in sync with code
