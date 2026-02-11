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
    start_age: int = 25
    retirement_age: int = 37
    expected_lifespan: int = 100

    # Starting conditions
    initial_net_worth: float = 500      # $500k
    lifestyle_inflation: float = 1.01   # 1% annual lifestyle creep

    # Income schedule ($1,000 units per year, pre-tax, inflation-adjusted)
    income_schedule: List[float] = field(default_factory=lambda:
        [350] * 4 + [450] * 8)

    # Expenses
    initial_expenses: float = 60        # $60k/year
    one_time_expenses: Dict[int, float] = field(default_factory=lambda: {
        28: 50,    # Wedding
        32: 300,   # House down payment
        **{a: 40 for a in range(25, 70, 5)},  # Car every 5 years
    })

    # Market return decomposition: E[stock] = bond_yield + ERP
    equity_risk_premium: float = 0.05   # 5% ERP (historical average)
    stock_vol: float = 0.10             # 10% annualized equity volatility (long-run mean)

    # Fat tails: Student-t degrees of freedom for stock return shocks
    # df=5 gives excess kurtosis ~6 (realistic crash frequency)
    # df=inf (or >100) reverts to normal distribution
    stock_tail_df: float = 5.0

    # Volatility clustering: log-normal stochastic volatility
    #   log(σ_t / σ_bar) = ρ * log(σ_{t-1} / σ_bar) + η * ε_vol
    # After a crash, vol stays elevated; in calm markets, vol is low
    vol_persistence: float = 0.80       # AR(1) persistence of vol state
    vol_of_vol: float = 0.25            # std of log-vol innovations

    # Stock-bond correlation (discount rate channel)
    # Negative: when bond yields spike up, stocks tend to fall
    stock_bond_corr: float = -0.20

    # Real bond yield dynamics (Vasicek mean-reverting model)
    #   dr = κ(θ - r)dt + σ_r dW
    initial_bond_yield: float = 0.02    # current real yield (~TIPS rate)
    long_run_bond_yield: float = 0.02   # long-run equilibrium θ
    bond_yield_mean_reversion: float = 0.15  # mean reversion speed κ
    bond_yield_vol: float = 0.012       # annual volatility of yield changes

    # Leverage costs: margin_fee = bond_yield + spread
    margin_spread: float = 0.015        # broker spread above risk-free rate

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

    # Utility scoring: U(c) = c^α, V = Σ β^t · U(spending_t)
    utility_power: float = 0.8         # CRRA exponent α
    discount_rate: float = 0.03        # annual time preference δ (β = 1/(1+δ))


@dataclass
class SimulationResult:
    """Results from a single simulation run."""
    portfolios: PortfolioManager
    spending: List[List[float]]  # spending[portfolio_idx][year_idx], $1k units


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
# CRRA Utility
# ---------------------------------------------------------------------------

def compute_lifetime_utility(
    spending: List[float],
    config: SimulationConfig,
) -> float:
    """Discounted CRRA utility of a lifetime spending stream.

    V = Σ_{t=0}^{T} β^t · spending_t^α
    where α = utility_power, β = 1/(1+discount_rate).
    """
    alpha = config.utility_power
    beta = 1.0 / (1.0 + config.discount_rate)
    total = 0.0
    for t, c in enumerate(spending):
        if c > 0:
            total += (beta ** t) * (c ** alpha)
    return total


def certainty_equivalent(
    utility: float,
    n_years: int,
    config: SimulationConfig,
) -> float:
    """Constant annual spending ($1k) giving the same lifetime utility.

    Solves: V = CE^α · Σ β^t  →  CE = (V / Σ β^t)^(1/α)
    """
    alpha = config.utility_power
    beta = 1.0 / (1.0 + config.discount_rate)
    if abs(beta - 1.0) < 1e-10:
        annuity = float(n_years)
    else:
        annuity = (1.0 - beta ** n_years) / (1.0 - beta)
    if annuity <= 0 or utility <= 0:
        return 0.0
    return (utility / annuity) ** (1.0 / alpha)


# ---------------------------------------------------------------------------
# Optimal Leverage (Kelly Criterion)
# ---------------------------------------------------------------------------

def compute_optimal_leverage(config: SimulationConfig) -> float:
    """Compute Kelly-optimal leverage ratio.

    Maximizes expected log growth rate of the leveraged portfolio:
        E[log(1 + r_lev)] ≈ E[r_lev] - Var(r_lev)/2
    where r_lev = L * r_stock - (L-1) * margin_fee.

    The excess return over borrowing cost = ERP - spread, which is
    independent of bond yield. Optimal leverage:
        L* = (ERP - spread) / σ²
    """
    excess = config.equity_risk_premium - config.margin_spread
    if excess <= 0:
        return 1.0  # No benefit to leverage if spread eats the ERP
    return excess / config.stock_vol ** 2


# ---------------------------------------------------------------------------
# Vasicek Bond Yield Model
# ---------------------------------------------------------------------------

def evolve_bond_yield(
    current_yield: float,
    config: SimulationConfig,
    bond_shock: float,
) -> float:
    """Advance the real bond yield by one year using the Vasicek model.

    dr = κ(θ - r)dt + σ_r * dW
    bond_shock is a standard normal draw (passed in so it can be
    correlated with stock shocks).
    Floor at -2% to prevent unrealistic extremes.
    """
    drift = config.bond_yield_mean_reversion * (config.long_run_bond_yield - current_yield)
    new_yield = current_yield + drift + config.bond_yield_vol * bond_shock
    return max(new_yield, -0.02)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_simulation(
    config: SimulationConfig | None = None,
    seed: int | None = None,
    quiet: bool = False,
) -> SimulationResult:
    """Run the retirement simulation and return results with spending history.

    Market model:
        bond_yield follows Vasicek mean-reverting process
        stock_return = bond_yield + ERP + σ_t * ε_stock
        margin_fee   = bond_yield + spread

    Realistic dynamics:
        - Fat tails: ε ~ Student-t(df) normalized to unit variance
        - Stochastic vol: log(σ_t/σ̄) = ρ·log(σ_{t-1}/σ̄) + η·ε_vol
        - Stock-bond correlation: ε_stock = corr·ε_bond + √(1-corr²)·ε_indep

    Spending is constrained: if a portfolio can't afford planned expenses,
    spending is reduced to what's available (NW + income, floored at 0).

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
        kelly = compute_optimal_leverage(config)
        print(f"Simulation seed: {seed}")
        print(f"E[stock return] = bond_yield + ERP = ~{config.initial_bond_yield + config.equity_risk_premium:.1%}")
        print(f"Margin fee = bond_yield + spread = ~{config.initial_bond_yield + config.margin_spread:.1%}")
        print(f"Kelly optimal leverage: {kelly:.2f}x  |  Half-Kelly: {kelly/2:.2f}x  |  Using: {config.leverage_ratio:.2f}x")
        tail_desc = f"Student-t(df={config.stock_tail_df:.0f})" if config.stock_tail_df <= 100 else "Normal"
        print(f"Return dist: {tail_desc}  |  Vol clustering: ρ={config.vol_persistence}, η={config.vol_of_vol}")
        print(f"Stock-bond corr: {config.stock_bond_corr:+.2f}")
        print()

    # Build income & expense arrays sized to simulation duration
    sim_years = config.expected_lifespan - config.start_age + 1
    realized_income = config.income_schedule[:config.retirement_age - config.start_age + 1]
    realized_income += [0] * (sim_years - len(realized_income))
    expenses = [config.initial_expenses * config.lifestyle_inflation ** i
                for i in range(sim_years)]

    # Mutable containers for closures shared with portfolio objects
    year_return = [1.0]
    year_margin_fee = [0.0]

    def real_mkt_return() -> float:
        return year_return[0]

    def margin_fee() -> float:
        return year_margin_fee[0]

    portfolios = PortfolioManager([
        CashPortfolio(config.initial_net_worth),
        StockPortfolio(config.initial_net_worth, real_mkt_return),
        LeveragedStockPortfolio(
            config.initial_net_worth, config.leverage_ratio,
            real_mkt_return, margin_fee,
        ),
    ])
    n_portfolios = len(portfolios.portfolios)
    spending_records: List[List[float]] = [[] for _ in range(n_portfolios)]

    bond_yield = config.initial_bond_yield
    log_vol = 0.0  # log(σ_t / σ_bar), starts at long-run mean

    df = config.stock_tail_df
    use_fat_tails = df <= 100 and df > 2
    # Normalization factor so Student-t shocks have unit variance
    t_scale = np.sqrt((df - 2) / df) if use_fat_tails else 1.0

    rho = config.stock_bond_corr
    rho_complement = np.sqrt(1 - rho ** 2)

    for i, curr_age in enumerate(range(config.start_age, config.expected_lifespan + 1)):
        # --- Generate correlated shocks ---
        # Bond yield shock (always normal — Vasicek model)
        bond_shock = rng.standard_normal()

        # Independent stock shock (fat-tailed if configured)
        if use_fat_tails:
            indep_shock = rng.standard_t(df) * t_scale
        else:
            indep_shock = rng.standard_normal()

        # Correlate: when bond yields spike, stocks tend to fall
        stock_shock = rho * bond_shock + rho_complement * indep_shock

        # Vol-of-vol shock (independent)
        vol_shock = rng.standard_normal()

        # --- Evolve stochastic volatility ---
        log_vol = config.vol_persistence * log_vol + config.vol_of_vol * vol_shock
        current_vol = config.stock_vol * np.exp(log_vol)

        # --- Evolve bond yield (Vasicek) ---
        bond_yield = evolve_bond_yield(bond_yield, config, bond_shock)

        # --- Compute returns ---
        # Stock return = bond_yield + ERP + σ_t * ε_stock
        year_return[0] = 1 + bond_yield + config.equity_risk_premium + current_vol * stock_shock

        # Margin fee = bond_yield + broker spread (floored at 0)
        year_margin_fee[0] = max(bond_yield + config.margin_spread, 0.0)

        inc = post_tax(realized_income[i], config)
        exp = expenses[i]
        if curr_age in config.one_time_expenses:
            exp += config.one_time_expenses[curr_age]

        portfolios.pass_year()

        # Constrained spending: each portfolio spends min(planned, available)
        for j, p in enumerate(portfolios.portfolios):
            available = p.get_nw() + inc
            actual_spend = max(0.0, min(exp, available))
            spending_records[j].append(actual_spend)
            p.add_money(inc)
            p.remove_money(actual_spend)

        if not quiet:
            print(f"Age {curr_age:3d}:  bond_yield={bond_yield:+.2%}  "
                  f"vol={current_vol:.1%}  margin_fee={year_margin_fee[0]:.2%}  "
                  f"income={format_money(inc)}  expenses={format_money(exp)}  "
                  f"nw={portfolios.get_nw()}")

    return SimulationResult(portfolios=portfolios, spending=spending_records)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(result: SimulationResult, config: SimulationConfig) -> None:
    """Plot net worth history for all portfolios over the simulation period."""
    portfolios = result.portfolios
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
        'mean_utility', 'mean_ce': expected utility metrics (theoretically correct)
        'median_utility', 'median_ce': robust utility metrics
    """
    all_histories: Dict[str, List[List[float]]] = {}
    all_utilities: Dict[str, List[float]] = {}

    sim_years = config.expected_lifespan - config.start_age + 1

    for sim in range(n_simulations):
        result = run_simulation(config, quiet=True)
        for j, p in enumerate(result.portfolios.portfolios):
            name = type(p).__name__
            if name not in all_histories:
                all_histories[name] = []
                all_utilities[name] = []
            all_histories[name].append(p.get_nw_history())
            all_utilities[name].append(
                compute_lifetime_utility(result.spending[j], config))

    first_key = next(iter(all_histories))
    ages = list(range(config.start_age,
                      config.start_age + len(all_histories[first_key][0])))

    results: Dict[str, Dict] = {}
    for name, histories in all_histories.items():
        arr = np.array(histories) / 1000  # Convert to $M
        u_arr = np.array(all_utilities[name])
        mean_u = float(np.mean(u_arr))
        median_u = float(np.median(u_arr))
        results[name] = {
            'ages': ages,
            'median': np.median(arr, axis=0),
            'p10': np.percentile(arr, 10, axis=0),
            'p25': np.percentile(arr, 25, axis=0),
            'p75': np.percentile(arr, 75, axis=0),
            'p90': np.percentile(arr, 90, axis=0),
            'mean_utility': mean_u,
            'mean_ce': certainty_equivalent(mean_u, sim_years, config),
            'median_utility': median_u,
            'median_ce': certainty_equivalent(median_u, sim_years, config),
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
        ce = data.get('mean_ce', data.get('median_ce', 0))
        ax.set_title(f'{name}\nE[CE]={format_money(ce)}/yr')
        ax.set_xlabel('Age')
        ax.legend(fontsize=8)

    axes[0].set_ylabel('Net Worth ($M)')
    fig.suptitle('Monte Carlo Retirement Simulation (U(c)=c^{:.1f})'.format(
        config.utility_power))
    plt.tight_layout()
    plt.show()

    # Print utility summary
    print(f"\n=== Utility Summary: U(c) = c^{config.utility_power}, "
          f"δ={config.discount_rate:.0%} ===")
    print(f"{'Portfolio':<28s}  {'E[Utility]':>10s}  {'E[CE]':>12s}  "
          f"{'Med[CE]':>12s}")
    print("-" * 68)
    for name, data in results.items():
        print(f"{name:<28s}  {data['mean_utility']:>10,.0f}  "
              f"{format_money(data['mean_ce']):>12s}/yr  "
              f"{format_money(data['median_ce']):>12s}/yr")
    print("(E[CE] = expected-utility CE; Med[CE] = median-path CE)")
    print()


# ---------------------------------------------------------------------------
# Leverage Sweep
# ---------------------------------------------------------------------------

def run_leverage_sweep(
    config: SimulationConfig,
    leverage_range: List[float] | None = None,
    n_simulations: int = 500,
) -> Dict[str, List]:
    """Sweep across leverage ratios and compute risk/reward statistics.

    For each leverage level, runs n_simulations Monte Carlo sims and
    tracks the LeveragedStockPortfolio (index 2) outcomes.

    Returns dict with keys: 'leverage', 'median_nw', 'p10_nw', 'p25_nw',
    'p75_nw', 'p90_nw', 'ruin_pct', 'mean_ce', 'median_ce'.
    """
    if leverage_range is None:
        leverage_range = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5]

    sim_years = config.expected_lifespan - config.start_age + 1

    results: Dict[str, List] = {
        'leverage': [], 'median_nw': [],
        'p10_nw': [], 'p25_nw': [], 'p75_nw': [], 'p90_nw': [],
        'ruin_pct': [], 'mean_ce': [], 'median_ce': [],
    }

    from dataclasses import replace

    for lev in leverage_range:
        cfg = replace(config, leverage_ratio=lev)
        final_nws = []
        utilities = []
        ruin_count = 0

        for _ in range(n_simulations):
            result = run_simulation(cfg, quiet=True)
            lev_portfolio = result.portfolios.portfolios[2]
            history = lev_portfolio.get_nw_history()
            final_nws.append(history[-1])
            if min(history) < 0:
                ruin_count += 1
            utilities.append(
                compute_lifetime_utility(result.spending[2], cfg))

        arr = np.array(final_nws) / 1000  # Convert to $M
        mean_u = float(np.mean(utilities))
        median_u = float(np.median(utilities))
        mean_ce = certainty_equivalent(mean_u, sim_years, cfg)
        median_ce = certainty_equivalent(median_u, sim_years, cfg)

        results['leverage'].append(lev)
        results['median_nw'].append(float(np.median(arr)))
        results['p10_nw'].append(float(np.percentile(arr, 10)))
        results['p25_nw'].append(float(np.percentile(arr, 25)))
        results['p75_nw'].append(float(np.percentile(arr, 75)))
        results['p90_nw'].append(float(np.percentile(arr, 90)))
        results['ruin_pct'].append(ruin_count / n_simulations)
        results['mean_ce'].append(mean_ce)
        results['median_ce'].append(median_ce)

        print(f"  {lev:.2f}x:  median=${results['median_nw'][-1]*1000:>10,.0f}k  "
              f"p10=${results['p10_nw'][-1]*1000:>10,.0f}k  "
              f"ruin={results['ruin_pct'][-1]:.1%}  "
              f"E[CE]={format_money(mean_ce)}/yr")

    return results


def plot_leverage_sweep(sweep: Dict[str, List], config: SimulationConfig) -> None:
    """Plot leverage sweep results: NW percentiles, ruin probability, and CE."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    kelly = compute_optimal_leverage(config)
    levs = sweep['leverage']

    # Left panel: Final NW percentiles vs leverage (log scale)
    ax1.fill_between(levs, sweep['p10_nw'], sweep['p90_nw'],
                     alpha=0.15, label='10th-90th %ile')
    ax1.fill_between(levs, sweep['p25_nw'], sweep['p75_nw'],
                     alpha=0.3, label='25th-75th %ile')
    ax1.plot(levs, sweep['median_nw'], 'o-', linewidth=2, label='Median')
    ax1.axvline(x=kelly, color='green', linestyle='--', alpha=0.7,
                label=f'Kelly optimal ({kelly:.2f}x)')
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Leverage Ratio')
    ax1.set_ylabel('Final Net Worth ($M)')
    ax1.set_title('Final NW vs Leverage')
    ax1.set_yscale('symlog', linthresh=1)
    ax1.legend(fontsize=8)

    # Center panel: Ruin probability vs leverage
    ax2.plot(levs, [r * 100 for r in sweep['ruin_pct']], 'o-',
             linewidth=2, color='red')
    ax2.axhline(y=5, color='orange', linestyle='--', alpha=0.7,
                label='5% risk threshold')
    ax2.axvline(x=kelly, color='green', linestyle='--', alpha=0.7,
                label=f'Kelly optimal ({kelly:.2f}x)')
    ax2.set_xlabel('Leverage Ratio')
    ax2.set_ylabel('Ruin Probability (%)')
    ax2.set_title('Ruin Risk vs Leverage')
    ax2.set_ylim(bottom=0)
    ax2.legend(fontsize=8)

    # Right panel: Certainty-equivalent spending vs leverage
    ce_vals = sweep['mean_ce']
    ax3.plot(levs, [c * 1000 for c in ce_vals], 'o-',
             linewidth=2, color='purple')
    ax3.axvline(x=kelly, color='green', linestyle='--', alpha=0.7,
                label=f'Kelly optimal ({kelly:.2f}x)')
    best_idx = int(np.argmax(ce_vals))
    ax3.axvline(x=levs[best_idx], color='purple', linestyle=':',
                alpha=0.7, label=f'Utility max ({levs[best_idx]:.2f}x)')
    ax3.set_xlabel('Leverage Ratio')
    ax3.set_ylabel('E[CE] Spending ($k/yr)')
    ax3.set_title(f'Expected CE (U=c^{config.utility_power})')
    ax3.legend(fontsize=8)

    fig.suptitle(f'Leverage Sweep (age {config.start_age}-{config.expected_lifespan}, '
                 f'retire {config.retirement_age})')
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Retirement financial planning simulator'
    )
    # Personal
    parser.add_argument('--start-age', type=int, default=25,
                        help='Age to start simulation (default: 25)')
    parser.add_argument('--retirement-age', type=int, default=37,
                        help='Retirement age (default: 37)')
    parser.add_argument('--lifespan', type=int, default=100,
                        help='Expected lifespan (default: 100)')
    parser.add_argument('--initial-nw', type=float, default=500,
                        help='Initial net worth in $1,000 units (default: 500)')
    parser.add_argument('--initial-expenses', type=float, default=60,
                        help='Initial annual expenses in $1,000 units (default: 60)')
    # Market model
    parser.add_argument('--erp', type=float, default=0.05,
                        help='Equity risk premium (default: 0.05)')
    parser.add_argument('--stock-vol', type=float, default=0.10,
                        help='Annualized equity volatility (default: 0.10)')
    parser.add_argument('--bond-yield', type=float, default=0.02,
                        help='Initial real bond yield (default: 0.02)')
    parser.add_argument('--margin-spread', type=float, default=0.015,
                        help='Broker spread above bond yield (default: 0.015)')
    # Leverage
    parser.add_argument('--leverage', type=float, default=2.0,
                        help='Leverage ratio for leveraged portfolio (default: 2.0)')
    parser.add_argument('--optimal-leverage', action='store_true',
                        help='Use Kelly-optimal leverage instead of --leverage')
    # Simulation mode
    parser.add_argument('--monte-carlo', type=int, default=0, metavar='N',
                        help='Run N Monte Carlo simulations (default: 0 = single run)')
    parser.add_argument('--leverage-sweep', type=int, default=0, metavar='N',
                        help='Run leverage sweep with N sims per level (default: 0 = off)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    # Utility
    parser.add_argument('--discount-rate', type=float, default=0.03,
                        help='Annual time discount rate for utility (default: 0.03)')
    parser.add_argument('--utility-power', type=float, default=0.8,
                        help='CRRA exponent: U(c) = c^α (default: 0.8)')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    leverage = args.leverage
    config = SimulationConfig(
        start_age=args.start_age,
        retirement_age=args.retirement_age,
        expected_lifespan=args.lifespan,
        initial_net_worth=args.initial_nw,
        initial_expenses=args.initial_expenses,
        equity_risk_premium=args.erp,
        stock_vol=args.stock_vol,
        initial_bond_yield=args.bond_yield,
        long_run_bond_yield=args.bond_yield,
        margin_spread=args.margin_spread,
        leverage_ratio=leverage,
        utility_power=args.utility_power,
        discount_rate=args.discount_rate,
    )

    if args.optimal_leverage:
        config.leverage_ratio = compute_optimal_leverage(config)
        print(f"Using Kelly-optimal leverage: {config.leverage_ratio:.2f}x")

    sim_years = config.expected_lifespan - config.start_age + 1

    if args.leverage_sweep > 0:
        print(f"Running leverage sweep ({args.leverage_sweep} sims per level)...")
        kelly = compute_optimal_leverage(config)
        print(f"Kelly optimal: {kelly:.2f}x  |  Half-Kelly: {kelly/2:.2f}x")
        print(f"Utility: U(c) = c^{config.utility_power}, "
              f"discount rate = {config.discount_rate:.0%}\n")
        sweep = run_leverage_sweep(config, n_simulations=args.leverage_sweep)
        plot_leverage_sweep(sweep, config)
    elif args.monte_carlo > 0:
        print(f"Running {args.monte_carlo} Monte Carlo simulations...")
        results = run_monte_carlo(config, n_simulations=args.monte_carlo)
        plot_monte_carlo(results, config)
    else:
        result = run_simulation(config, seed=args.seed)
        # Display utility scores
        print(f"\n=== Lifetime Utility (U(c) = c^{config.utility_power}, "
              f"δ={config.discount_rate:.0%}) ===")
        print(f"{'Portfolio':<30s}  {'Utility':>10s}  {'CE Spending':>14s}")
        print("-" * 58)
        for j, p in enumerate(result.portfolios.portfolios):
            u = compute_lifetime_utility(result.spending[j], config)
            ce = certainty_equivalent(u, sim_years, config)
            print(f"{type(p).__name__:<30s}  {u:>10,.0f}  {format_money(ce):>14s}/yr")
        print()
        plot_results(result, config)


if __name__ == '__main__':
    main()
