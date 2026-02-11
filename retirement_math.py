# Retirement financial planning simulator
# All monetary values are in present-dollar terms, 1 unit = $1,000 (2023 USD)

import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from portfolio import (
    CashPortfolio,
    LeveragedStockPortfolio,
    Portfolio,
    PortfolioManager,
    StockPortfolio,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    """All tunable parameters for the retirement simulation.

    Monetary values are in $1,000 units (2023 USD).
    """

    # Personal timeline
    start_age: int = 22
    retirement_age: int = 65
    expected_lifespan: int = 100

    # Starting conditions
    initial_net_worth: float = 40       # $40k
    lifestyle_inflation: float = 1.01   # 1% annual lifestyle creep

    # Income schedule ($1,000 units per year, pre-tax)
    income_schedule: List[float] = field(default_factory=lambda:
        [230] * 3 + [260] * 3 + [400] * 4 + [550] * 68)

    # Expenses
    initial_expenses: float = 60        # $60k/year
    one_time_expenses: Dict[int, float] = field(default_factory=lambda: {
        28: 50,    # Wedding
        32: 300,   # House down payment
        **{a: 40 for a in range(25, 70, 5)},  # Car every 5 years
    })

    # Market assumptions
    real_return_mean: float = 0.035     # 3.5% real return
    real_return_std: float = 0.1        # 10% volatility

    # Tax: progressive federal brackets (2023) + flat state rate
    state_tax_rate: float = 0.15
    federal_brackets: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0,        0.00),
        (9_950,    0.10),
        (40_525,   0.12),
        (86_375,   0.22),
        (164_925,  0.24),
        (209_425,  0.32),
        (523_600,  0.35),
        (10**10,   0.37),
    ])

    # Leverage for leveraged portfolio
    leverage_ratio: float = 2.0


# ---------------------------------------------------------------------------
# Tax & Formatting Helpers
# ---------------------------------------------------------------------------

def post_tax(gross_income: float, config: SimulationConfig) -> float:
    """Compute post-tax income given gross income in $1,000 units.

    Applies progressive federal tax brackets and a flat state tax rate.
    Returns post-tax income in $1,000 units.
    """
    gross = gross_income * 1000  # Convert to dollar units for bracket math
    brackets = config.federal_brackets  # Already sorted

    federal_tax = 0.0
    remaining = gross
    for i in range(len(brackets) - 1):
        bracket_size = brackets[i + 1][0] - brackets[i][0]
        if remaining <= bracket_size:
            federal_tax += remaining * brackets[i][1]
            break
        federal_tax += bracket_size * brackets[i][1]
        remaining -= bracket_size

    state_tax = gross * config.state_tax_rate
    return (gross - federal_tax - state_tax) / 1000


def format_money(val: float) -> str:
    """Format a $1,000-unit value as a human-readable dollar string."""
    return '${:,}'.format(int(val * 1000))


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_simulation(
    config: SimulationConfig | None = None,
    seed: int | None = None,
    quiet: bool = False,
) -> PortfolioManager:
    """Run the retirement simulation and return the PortfolioManager.

    Args:
        config: Simulation parameters (uses defaults if None).
        seed: Random seed for reproducibility (random if None).
        quiet: Suppress per-year print output when True.
    """
    if config is None:
        config = SimulationConfig()
    if seed is None:
        seed = np.random.randint(0, 1_000_000)

    rng = np.random.default_rng(seed)
    if not quiet:
        print(f"Simulation seed: {seed}")

    # Build income & expense arrays sized to simulation duration
    sim_years = config.expected_lifespan - config.start_age + 1
    realized_income = config.income_schedule[:config.retirement_age - config.start_age + 1]
    realized_income += [0] * (sim_years - len(realized_income))
    expenses = [config.initial_expenses * config.lifestyle_inflation ** i
                for i in range(sim_years)]

    # Market return shared across all portfolios each year
    year_return = [1.0]

    def real_mkt_return() -> float:
        return year_return[0]

    portfolios = PortfolioManager([
        CashPortfolio(config.initial_net_worth),
        StockPortfolio(config.initial_net_worth, real_mkt_return),
        LeveragedStockPortfolio(
            config.initial_net_worth, config.leverage_ratio, real_mkt_return
        ),
    ])

    for i, curr_age in enumerate(range(config.start_age, config.expected_lifespan + 1)):
        year_return[0] = 1 + rng.normal(config.real_return_mean, config.real_return_std)

        inc = post_tax(realized_income[i], config)
        exp = expenses[i]
        if curr_age in config.one_time_expenses:
            exp += config.one_time_expenses[curr_age]

        portfolios.pass_year()
        portfolios.remove_money(exp)
        portfolios.add_money(inc)

        if not quiet:
            print(f"Age {curr_age:3d}:  income={format_money(inc)}  "
                  f"expenses={format_money(exp)}  nw={portfolios.get_nw()}")

    return portfolios


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(portfolios: PortfolioManager, config: SimulationConfig) -> None:
    """Plot net worth history for all portfolios over the simulation period."""
    histories = portfolios.get_nw_history()
    ages = list(range(config.start_age, config.start_age + len(histories[0])))

    for i, portfolio in enumerate(portfolios.portfolios):
        values = [h / 1000 for h in histories[i]]
        plt.plot(ages, values, label=type(portfolio).__name__)

    plt.legend()
    plt.xlabel('Age')
    plt.ylabel('Net Worth ($M)')
    plt.axvline(x=config.retirement_age, color='r', linestyle='--')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Retirement Portfolio Simulation')
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

def run_monte_carlo(
    config: SimulationConfig,
    n_simulations: int = 500,
) -> Dict[str, Dict]:
    """Run multiple simulations and return percentile statistics.

    Returns a dict mapping portfolio class names to:
        'ages': list of ages
        'median', 'p10', 'p25', 'p75', 'p90': arrays of NW in $M
    """
    all_histories: Dict[str, List[List[float]]] = {}

    for sim in range(n_simulations):
        pm = run_simulation(config, quiet=True)
        for p in pm.portfolios:
            name = type(p).__name__
            if name not in all_histories:
                all_histories[name] = []
            all_histories[name].append(p.get_nw_history())

    first_key = next(iter(all_histories))
    ages = list(range(config.start_age,
                      config.start_age + len(all_histories[first_key][0])))

    results: Dict[str, Dict] = {}
    for name, histories in all_histories.items():
        arr = np.array(histories) / 1000  # Convert to $M
        results[name] = {
            'ages': ages,
            'median': np.median(arr, axis=0),
            'p10': np.percentile(arr, 10, axis=0),
            'p25': np.percentile(arr, 25, axis=0),
            'p75': np.percentile(arr, 75, axis=0),
            'p90': np.percentile(arr, 90, axis=0),
        }

    return results


def plot_monte_carlo(results: Dict[str, Dict], config: SimulationConfig) -> None:
    """Plot Monte Carlo results with shaded confidence bands."""
    n_portfolios = len(results)
    fig, axes = plt.subplots(1, n_portfolios,
                             figsize=(6 * n_portfolios, 5), sharey=True)
    if n_portfolios == 1:
        axes = [axes]

    for ax, (name, data) in zip(axes, results.items()):
        ages = data['ages']
        ax.fill_between(ages, data['p10'], data['p90'],
                        alpha=0.15, label='10th-90th %ile')
        ax.fill_between(ages, data['p25'], data['p75'],
                        alpha=0.3, label='25th-75th %ile')
        ax.plot(ages, data['median'], linewidth=2, label='Median')
        ax.axvline(x=config.retirement_age, color='r', linestyle='--', alpha=0.5)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.set_title(name)
        ax.set_xlabel('Age')
        ax.legend(fontsize=8)

    axes[0].set_ylabel('Net Worth ($M)')
    fig.suptitle('Monte Carlo Retirement Simulation')
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Retirement financial planning simulator'
    )
    parser.add_argument('--start-age', type=int, default=22,
                        help='Age to start simulation (default: 22)')
    parser.add_argument('--retirement-age', type=int, default=65,
                        help='Retirement age (default: 65)')
    parser.add_argument('--lifespan', type=int, default=100,
                        help='Expected lifespan (default: 100)')
    parser.add_argument('--initial-nw', type=float, default=40,
                        help='Initial net worth in $1,000 units (default: 40)')
    parser.add_argument('--initial-expenses', type=float, default=60,
                        help='Initial annual expenses in $1,000 units (default: 60)')
    parser.add_argument('--leverage', type=float, default=2.0,
                        help='Leverage ratio for leveraged portfolio (default: 2.0)')
    parser.add_argument('--monte-carlo', type=int, default=0, metavar='N',
                        help='Run N Monte Carlo simulations (default: 0 = single run)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SimulationConfig(
        start_age=args.start_age,
        retirement_age=args.retirement_age,
        expected_lifespan=args.lifespan,
        initial_net_worth=args.initial_nw,
        initial_expenses=args.initial_expenses,
        leverage_ratio=args.leverage,
    )

    if args.monte_carlo > 0:
        print(f"Running {args.monte_carlo} Monte Carlo simulations...")
        results = run_monte_carlo(config, n_simulations=args.monte_carlo)
        plot_monte_carlo(results, config)
    else:
        portfolios = run_simulation(config, seed=args.seed)
        plot_results(portfolios, config)


if __name__ == '__main__':
    main()
