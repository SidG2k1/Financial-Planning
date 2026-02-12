# Financial-Planning

A Monte Carlo retirement simulator that answers: **"When can I retire, and how much can I spend?"**

Compares three portfolio strategies — cash, stocks, and leveraged stocks — over a configurable lifetime, using lifecycle spending rules that adapt to market conditions, vitality (health/energy decline with age), and Social Security.

Heavily inspired by the principles of Ben Felix.

## Architecture

The simulator has three composable layers:

1. **Financial Model** — stochastic market simulation (Vasicek bond yields, equity returns with fat tails, stochastic volatility, stock-bond correlation)
2. **Decision Model** — spending rules that determine how much to consume each year (fixed, lifecycle amortized, or vitality-weighted amortized)
3. **Utility Model** — CRRA power utility with FIRE multiplier and vitality weighting, used to score and compare outcomes

## Quick Start

```bash
pip install -r requirements.txt

# Single simulation
python retirement_math.py

# Monte Carlo (500 sims, plots confidence intervals)
python retirement_math.py --monte-carlo 500

# Find optimal retirement age (the main use case)
python retirement_math.py --retirement-sweep 500
```

## Key Concepts

### Vitality Curve
Health and energy decline with age. The vitality function `v(age)` models this as a Gaussian decay from a peak age, with a floor representing baseline capacity. This causes the spending rule to **front-load consumption into high-vitality years** — spend more at 35, less at 80.

### FIRE Multiplier
Retirement years are worth more than working years (freedom, autonomy). The FIRE multiplier inflates utility during retirement: `U = fire_mult * v(age) * c^α`. Higher values pull the optimal retirement age earlier.

### Social Security
SS benefits are computed from the full formula (AIME over top 35 earning years, PIA bend points, FRA adjustment). **FIRE reduces SS** — retiring after 12 years of work yields ~$18k/yr vs ~$32k/yr for 35+ years. The spending rules include PV of future SS when planning.

### Spending Floor
The vitality-weighted rule can front-load spending aggressively, leaving little for old age (especially with leveraged portfolios). The spending floor reserves resources for a guaranteed minimum: `spending = floor + excess * v(age) / v_annuity`. Default $45k/yr.

### Retirement Tax Advantage
Retirement withdrawals (LTCG at ~15%, Roth at 0%) are taxed much less than earned income (~38%). The `retirement_tax_advantage` multiplier (default 1.25x) inflates the real purchasing power of portfolio wealth after retirement.

## Simulation Modes

### Single Run
```bash
python retirement_math.py --seed 42
```
One simulation path with year-by-year output. Good for understanding the mechanics.

### Monte Carlo
```bash
python retirement_math.py --monte-carlo 1000
```
Runs N simulations, plots net worth trajectories with confidence bands for each portfolio type.

### Retirement Age Sweep
```bash
python retirement_math.py --retirement-sweep 500
```
The main optimization mode. Sweeps retirement ages from 30-70, runs N sims per age, and plots certainty-equivalent (CE) spending vs retirement age. Automatically uses vitality-weighted amortized spending. Shows the optimal retirement age per portfolio type.

### Leverage Sweep
```bash
python retirement_math.py --leverage-sweep 500
```
Sweeps leverage ratios from 1x-3x to find the optimal leverage level.

## CLI Reference

### Personal / Timeline
| Flag | Default | Description |
|------|---------|-------------|
| `--start-age` | 25 | Age to start simulation |
| `--retirement-age` | 37 | Retirement age |
| `--lifespan` | 90 | Expected lifespan |
| `--initial-nw` | 500 | Initial net worth ($1k units) |
| `--initial-expenses` | 60 | Initial annual expenses ($1k units) |
| `--spending-floor` | 45 | Minimum annual spending ($1k units) |

### Market Model
| Flag | Default | Description |
|------|---------|-------------|
| `--erp` | 0.05 | Equity risk premium |
| `--stock-vol` | 0.10 | Annualized equity volatility |
| `--bond-yield` | 0.02 | Initial real bond yield (≈ TIPS rate) |
| `--margin-spread` | 0.015 | Broker spread above bond yield for leverage |

### Portfolio
| Flag | Default | Description |
|------|---------|-------------|
| `--leverage` | 2.0 | Leverage ratio for leveraged portfolio |
| `--optimal-leverage` | — | Use Kelly-optimal leverage instead |

### Simulation Mode
| Flag | Default | Description |
|------|---------|-------------|
| `--monte-carlo N` | 0 | Run N Monte Carlo sims (0 = single run) |
| `--retirement-sweep N` | 0 | Sweep retirement ages with N sims per age |
| `--leverage-sweep N` | 0 | Sweep leverage ratios with N sims per level |
| `--seed` | random | Random seed for reproducibility |

### Spending Rule
| Flag | Default | Description |
|------|---------|-------------|
| `--amortized` | — | Use lifecycle amortized spending (auto for sweeps) |

### Utility Model
| Flag | Default | Description |
|------|---------|-------------|
| `--utility-power` | 0.65 | CRRA exponent α in U(c) = c^α |
| `--discount-rate` | 0.03 | Annual time preference δ |
| `--fire-multiplier` | 1.8 | Utility multiplier for retirement years |

### Vitality
| Flag | Default | Description |
|------|---------|-------------|
| `--vitality-peak` | 30 | Age of peak vitality |
| `--vitality-half-life` | 35 | Years from peak to ~63% decay |
| `--vitality-floor` | 0.3 | Minimum vitality (even at 100) |
| `--no-vitality` | — | Disable vitality weighting |

### Social Security
| Flag | Default | Description |
|------|---------|-------------|
| `--no-ss` | — | Disable Social Security |
| `--ss-claiming-age` | 67 | Age to start claiming (62-70) |

### Tax
| Flag | Default | Description |
|------|---------|-------------|
| `--retirement-tax-advantage` | 1.25 | Withdrawal purchasing power multiplier |

## Example Workflows

### "When should I retire?"
```bash
# Default high-income profile ($350-450k, $500k NW at 25)
python retirement_math.py --retirement-sweep 500

# With strong FIRE preference
python retirement_math.py --retirement-sweep 500 --fire-multiplier 2.5

# Conservative: no leverage, no SS
python retirement_math.py --retirement-sweep 500 --leverage 1.0 --no-ss
```

### "How much can I spend at each age?"
Run a single sim with amortized spending and inspect the year-by-year output:
```bash
python retirement_math.py --amortized --seed 42
```

### "What leverage should I use?"
```bash
python retirement_math.py --leverage-sweep 500
# Or use Kelly-optimal:
python retirement_math.py --optimal-leverage
```

## Portfolio Strategies

- **CashPortfolio** — earns the real bond yield (≈2%). Safe but low growth.
- **StockPortfolio** — earns bond yield + ERP (≈7%) with stochastic volatility, fat tails, and stock-bond correlation.
- **LeveragedStockPortfolio** — leveraged equities with borrowing costs. Default 2x. Higher expected returns but much higher variance; volatility drag reduces median growth below mean growth.

## Monetary Units

All values are in **$1,000 units** (2023 USD). An income of `350` means $350,000/year. Output is displayed in full dollar amounts.

## Contributing

1. Fork the repo and create a feature branch
2. The codebase is two files: `retirement_math.py` (simulation + CLI) and `portfolio.py` (portfolio classes)
3. Key design principles:
   - All monetary values in $1k units
   - Three-layer architecture: Financial Model, Decision Model, Utility Model — keep them decoupled
   - Spending rules implement `SpendingRule.compute(ctx: YearContext) -> float`
   - Utility scorers implement `UtilityScorer.score()` and `certainty_equivalent()`
   - New features should be configurable via `SimulationConfig` fields and CLI flags
4. Test changes with: `python retirement_math.py --retirement-sweep 200` (fast sanity check)
5. Open a PR with a description of what changed and why
