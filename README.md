# Financial-Planning

A financial simulator for personal retirement planning. Compares three investment strategies — cash, stocks, and leveraged stocks — over a configurable lifetime.

Heavily inspired by the principles of Ben Felix.

## Quick Start

```bash
pip install -r requirements.txt
python retirement_math.py
```

## Usage

### Single Simulation (default)

```bash
python retirement_math.py
python retirement_math.py --seed 42                  # Reproducible run
python retirement_math.py --retirement-age 60        # Retire at 60
python retirement_math.py --leverage 1.5 --initial-nw 100
```

### Monte Carlo Simulation

Run hundreds of simulations to see confidence intervals instead of a single random outcome:

```bash
python retirement_math.py --monte-carlo 500
python retirement_math.py --monte-carlo 1000 --retirement-age 55
```

### All CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--start-age` | 22 | Age to start simulation |
| `--retirement-age` | 65 | Retirement age |
| `--lifespan` | 100 | Expected lifespan |
| `--initial-nw` | 40 | Initial net worth ($1,000 units) |
| `--initial-expenses` | 60 | Initial annual expenses ($1,000 units) |
| `--leverage` | 2.0 | Leverage ratio for leveraged portfolio |
| `--monte-carlo N` | 0 | Run N Monte Carlo simulations (0 = single run) |
| `--seed` | random | Random seed for reproducibility |

## Portfolio Strategies

- **CashPortfolio**: Holds cash, loses value to 2% annual inflation
- **StockPortfolio**: Equity portfolio growing at real market returns (default: 3.5% mean, 10% vol)
- **LeveragedStockPortfolio**: Leveraged equities with borrowing costs (default: 2x leverage, 2% borrow rate)

## Sample Simulation

![Figure_1](https://user-images.githubusercontent.com/32756129/221748255-7fb2a464-c954-4260-9c6f-a9babaeb1c4f.png)
