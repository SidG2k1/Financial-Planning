# Retirement financial planning simulator
# All monetary values are in present-dollar terms, 1 unit = $1,000 (2023 USD)
#
# Architecture: Financial Model (simulation) + Utility Model (scoring)
#               + Decision Model (spending rules, sweeps)

import argparse
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Tuple

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
    stock_tail_df: float = 5.0

    # Volatility clustering: log-normal stochastic volatility
    vol_persistence: float = 0.80       # AR(1) persistence of vol state
    vol_of_vol: float = 0.25            # std of log-vol innovations

    # Stock-bond correlation (discount rate channel)
    stock_bond_corr: float = -0.20

    # Real bond yield dynamics (Vasicek mean-reverting model)
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

    # Utility parameters (used by CRRAUtility, kept here for CLI convenience)
    utility_power: float = 0.8         # CRRA exponent α
    discount_rate: float = 0.03        # annual time preference δ (β = 1/(1+δ))
    fire_multiplier: float = 1.0       # utility mult for retirement years (1.5 = FIRE)

    # Vitality curve: v(age) = floor + (1-floor)*exp(-((age-peak)/half_life)^2)
    # Captures declining health/energy/capacity to enjoy spending with age.
    # Based on QALY literature: ~1.0 at 30, ~0.80 at 50, ~0.55 at 65, ~0.4 at 80.
    vitality_peak_age: int = 30        # age of peak vitality
    vitality_half_life: float = 35.0   # age offset for ~63% decay of (1-floor)
    vitality_floor: float = 0.3        # minimum vitality (even at 100)


@dataclass
class SimulationResult:
    """Results from a single simulation run."""
    portfolios: PortfolioManager
    spending: List[List[float]]  # spending[portfolio_idx][year_idx], $1k units


# ---------------------------------------------------------------------------
# Tax & Formatting Helpers
# ---------------------------------------------------------------------------

def post_tax(gross_income: float, config: SimulationConfig) -> float:
    """Compute post-tax income given gross income in $1,000 units."""
    gross = gross_income * 1000
    brackets = config.federal_brackets

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


def vitality_at_age(age: int, config: SimulationConfig) -> float:
    """Vitality multiplier at a given age.

    Gaussian decay from peak:
        v(age) = floor + (1 - floor) * exp(-((age - peak) / half_life)^2)
    For age <= peak, returns 1.0.
    """
    if age <= config.vitality_peak_age:
        return 1.0
    x = (age - config.vitality_peak_age) / config.vitality_half_life
    return config.vitality_floor + (1.0 - config.vitality_floor) * np.exp(-(x * x))


# ===========================================================================
# DECISION MODEL: Spending Rules
# ===========================================================================

@dataclass
class YearContext:
    """All info a spending rule needs for one year's decision."""
    nw: float               # portfolio NW after returns
    income: float           # post-tax income this year ($1k)
    age: int
    retirement_age: int
    expected_lifespan: int
    bond_yield: float       # current real bond yield
    equity_risk_premium: float
    one_time_expense: float # one-time expense this year ($1k), 0 if none
    income_schedule: List[float]  # full pre-tax income schedule
    year_idx: int           # 0-based sim year index
    start_age: int
    config: SimulationConfig


class SpendingRule(ABC):
    """Determines how much to spend each year."""

    @abstractmethod
    def compute(self, ctx: YearContext) -> float:
        """Return target spending ($1k) for this year."""
        ...


class FixedSpending(SpendingRule):
    """Fixed expenses growing at lifestyle inflation, plus one-time events."""

    def __init__(self, initial_expenses: float, lifestyle_inflation: float) -> None:
        self.initial_expenses = initial_expenses
        self.lifestyle_inflation = lifestyle_inflation

    def compute(self, ctx: YearContext) -> float:
        year = ctx.age - ctx.start_age
        base = self.initial_expenses * self.lifestyle_inflation ** year
        return base + ctx.one_time_expense


class AmortizedSpending(SpendingRule):
    """Lifecycle spending: annuitize total resources over remaining lifespan.

    Each year recomputes sustainable spending based on:
        total_resources = NW + income + PV(future_income) - PV(future_one_time)
        regular_spending = total_resources / annuity_factor(remaining, r)
    where r = bond_yield + ERP (expected real return, adapts to rate env).
    """

    def compute(self, ctx: YearContext) -> float:
        remaining = ctx.expected_lifespan - ctx.age + 1
        if remaining <= 0:
            return 0.0

        r = max(ctx.bond_yield + ctx.equity_risk_premium, 0.001)

        # PV of future income (next year onward until retirement)
        working_years_left = max(0, ctx.retirement_age - ctx.age)
        d = 1.0 / (1.0 + r)
        pv_income = 0.0
        for s in range(1, working_years_left + 1):
            future_idx = ctx.year_idx + s
            if future_idx < len(ctx.income_schedule):
                gross = ctx.income_schedule[future_idx]
            elif ctx.income_schedule:
                gross = ctx.income_schedule[-1]
            else:
                gross = 0.0
            pv_income += post_tax(gross, ctx.config) * (d ** s)

        # PV of future one-time expenses (next year onward)
        pv_onetime = 0.0
        for future_age, amount in ctx.config.one_time_expenses.items():
            if future_age > ctx.age:
                pv_onetime += amount * (d ** (future_age - ctx.age))

        # Total resources for regular spending
        total = (ctx.nw + ctx.income + pv_income
                 - pv_onetime - ctx.one_time_expense)

        # Annuity factor: PV of $1/year for `remaining` years at rate r
        annuity = (1.0 - d ** remaining) / (1.0 - d)

        regular = max(0.0, total / annuity)
        return regular + ctx.one_time_expense


# ===========================================================================
# UTILITY MODEL: Scoring
# ===========================================================================

class UtilityScorer(ABC):
    """Scores a lifetime spending stream."""

    @abstractmethod
    def score(self, spending: List[float], config: SimulationConfig) -> float:
        """Compute lifetime utility of a spending stream."""
        ...

    @abstractmethod
    def certainty_equivalent(self, utility: float, n_years: int) -> float:
        """Constant annual spending giving the same lifetime utility."""
        ...


class CRRAUtility(UtilityScorer):
    """CRRA power utility with FIRE multiplier and vitality weighting.

    V = Σ β^t · vitality(age_t) · fire_mult_t · c_t^α

    Vitality captures declining health/energy with age (QALY-inspired).
    FIRE multiplier captures extra value of freedom when retired.
    """

    def __init__(self, power: float = 0.8, discount_rate: float = 0.03,
                 fire_multiplier: float = 1.0,
                 retirement_year_idx: int | None = None,
                 start_age: int = 25,
                 config: SimulationConfig | None = None) -> None:
        self.power = power
        self.discount_rate = discount_rate
        self.fire_multiplier = fire_multiplier
        self.retirement_year_idx = retirement_year_idx
        self.start_age = start_age
        self._config = config or SimulationConfig()

    def _weight(self, t: int) -> float:
        """Compute weight for year t: β^t · vitality · fire_mult."""
        beta = 1.0 / (1.0 + self.discount_rate)
        age = self.start_age + t
        v = vitality_at_age(age, self._config)
        if (self.retirement_year_idx is not None
                and t > self.retirement_year_idx):
            m = self.fire_multiplier
        else:
            m = 1.0
        return (beta ** t) * v * m

    def score(self, spending: List[float], config: SimulationConfig) -> float:
        total = 0.0
        for t, c in enumerate(spending):
            if c > 0:
                total += self._weight(t) * (c ** self.power)
        return total

    def certainty_equivalent(self, utility: float, n_years: int) -> float:
        """CE: constant spending giving same lifetime utility.

        Solves: Σ w_t · CE^α = V  →  CE = (V / Σ w_t)^(1/α)
        where w_t = β^t · vitality(age_t) · fire_mult_t.
        """
        w_sum = sum(self._weight(t) for t in range(n_years))
        if w_sum <= 0 or utility <= 0:
            return 0.0
        return (utility / w_sum) ** (1.0 / self.power)

    @classmethod
    def from_config(cls, config: SimulationConfig) -> 'CRRAUtility':
        """Build from SimulationConfig fields."""
        return cls(
            power=config.utility_power,
            discount_rate=config.discount_rate,
            fire_multiplier=config.fire_multiplier,
            retirement_year_idx=config.retirement_age - config.start_age,
            start_age=config.start_age,
            config=config,
        )


# ---------------------------------------------------------------------------
# Optimal Leverage (Kelly Criterion)
# ---------------------------------------------------------------------------

def compute_optimal_leverage(config: SimulationConfig) -> float:
    """Compute Kelly-optimal leverage: L* = (ERP - spread) / σ²."""
    excess = config.equity_risk_premium - config.margin_spread
    if excess <= 0:
        return 1.0
    return excess / config.stock_vol ** 2


# ---------------------------------------------------------------------------
# Vasicek Bond Yield Model
# ---------------------------------------------------------------------------

def evolve_bond_yield(
    current_yield: float,
    config: SimulationConfig,
    bond_shock: float,
) -> float:
    """Advance real bond yield by one year. Floor at -2%."""
    drift = config.bond_yield_mean_reversion * (config.long_run_bond_yield - current_yield)
    new_yield = current_yield + drift + config.bond_yield_vol * bond_shock
    return max(new_yield, -0.02)


# ===========================================================================
# FINANCIAL MODEL: Simulation Engine
# ===========================================================================

def _build_income_schedule(config: SimulationConfig) -> Tuple[List[float], List[float]]:
    """Build realized income array and extended schedule for PV calculations.

    If retirement_age exceeds the income_schedule length, extends with the
    last salary value (working longer = still earning).

    Returns:
        (realized_income, extended_schedule) both sized to sim_years.
    """
    sim_years = config.expected_lifespan - config.start_age + 1
    working_years = config.retirement_age - config.start_age

    # Extended schedule: pad with last value if working beyond schedule
    extended = list(config.income_schedule)
    if len(extended) < working_years and extended:
        extended += [extended[-1]] * (working_years - len(extended))

    # Realized: income during working years, then 0
    realized = list(extended[:working_years])
    realized += [0] * (sim_years - len(realized))

    # Extended schedule also padded to full sim_years for PV lookups
    extended_full = list(extended)
    extended_full += [0] * (sim_years - len(extended_full))

    return realized, extended_full


def run_simulation(
    config: SimulationConfig | None = None,
    spending_rule: SpendingRule | None = None,
    seed: int | None = None,
    quiet: bool = False,
) -> SimulationResult:
    """Run the retirement simulation with pluggable spending rule.

    Market model:
        bond_yield follows Vasicek mean-reverting process
        stock_return = bond_yield + ERP + σ_t * ε_stock
        margin_fee   = bond_yield + spread

    Realistic dynamics:
        - Fat tails: ε ~ Student-t(df) normalized to unit variance
        - Stochastic vol: log(σ_t/σ̄) = ρ·log(σ_{t-1}/σ̄) + η·ε_vol
        - Stock-bond correlation: ε_stock = corr·ε_bond + √(1-corr²)·ε_indep

    Spending is constrained: if a portfolio can't afford target spending,
    it's reduced to what's available (NW + income, floored at 0).
    """
    if config is None:
        config = SimulationConfig()
    if spending_rule is None:
        spending_rule = FixedSpending(config.initial_expenses, config.lifestyle_inflation)
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
        print(f"Spending: {type(spending_rule).__name__}")
        print()

    sim_years = config.expected_lifespan - config.start_age + 1
    realized_income, extended_schedule = _build_income_schedule(config)

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
    log_vol = 0.0

    df = config.stock_tail_df
    use_fat_tails = df <= 100 and df > 2
    t_scale = np.sqrt((df - 2) / df) if use_fat_tails else 1.0

    rho = config.stock_bond_corr
    rho_complement = np.sqrt(1 - rho ** 2)

    for i, curr_age in enumerate(range(config.start_age, config.expected_lifespan + 1)):
        # --- Generate correlated shocks ---
        bond_shock = rng.standard_normal()
        if use_fat_tails:
            indep_shock = rng.standard_t(df) * t_scale
        else:
            indep_shock = rng.standard_normal()
        stock_shock = rho * bond_shock + rho_complement * indep_shock
        vol_shock = rng.standard_normal()

        # --- Evolve stochastic volatility ---
        log_vol = config.vol_persistence * log_vol + config.vol_of_vol * vol_shock
        current_vol = config.stock_vol * np.exp(log_vol)

        # --- Evolve bond yield (Vasicek) ---
        bond_yield = evolve_bond_yield(bond_yield, config, bond_shock)

        # --- Compute returns ---
        year_return[0] = 1 + bond_yield + config.equity_risk_premium + current_vol * stock_shock
        year_margin_fee[0] = max(bond_yield + config.margin_spread, 0.0)

        inc = post_tax(realized_income[i], config)
        one_time = config.one_time_expenses.get(curr_age, 0.0)

        portfolios.pass_year()

        # Per-portfolio spending via the spending rule
        for j, p in enumerate(portfolios.portfolios):
            ctx = YearContext(
                nw=p.get_nw(), income=inc, age=curr_age,
                retirement_age=config.retirement_age,
                expected_lifespan=config.expected_lifespan,
                bond_yield=bond_yield,
                equity_risk_premium=config.equity_risk_premium,
                one_time_expense=one_time,
                income_schedule=extended_schedule,
                year_idx=i, start_age=config.start_age, config=config,
            )
            target = spending_rule.compute(ctx)
            available = p.get_nw() + inc
            actual = max(0.0, min(target, available))
            spending_records[j].append(actual)
            p.add_money(inc)
            p.remove_money(actual)

        if not quiet:
            avg_spend = sum(spending_records[j_][-1]
                            for j_ in range(n_portfolios)) / n_portfolios
            print(f"Age {curr_age:3d}:  bond_yield={bond_yield:+.2%}  "
                  f"vol={current_vol:.1%}  margin_fee={year_margin_fee[0]:.2%}  "
                  f"income={format_money(inc)}  spend~{format_money(avg_spend)}  "
                  f"nw={portfolios.get_nw()}")

    return SimulationResult(portfolios=portfolios, spending=spending_records)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(result: SimulationResult, config: SimulationConfig) -> None:
    """Plot net worth history for all portfolios."""
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
    spending_rule: SpendingRule | None = None,
    utility_scorer: UtilityScorer | None = None,
    n_simulations: int = 500,
) -> Dict[str, Dict]:
    """Run multiple simulations and return percentile statistics + utility."""
    if spending_rule is None:
        spending_rule = FixedSpending(config.initial_expenses, config.lifestyle_inflation)
    if utility_scorer is None:
        utility_scorer = CRRAUtility.from_config(config)

    all_histories: Dict[str, List[List[float]]] = {}
    all_utilities: Dict[str, List[float]] = {}
    sim_years = config.expected_lifespan - config.start_age + 1

    for sim in range(n_simulations):
        result = run_simulation(config, spending_rule, quiet=True)
        for j, p in enumerate(result.portfolios.portfolios):
            name = type(p).__name__
            if name not in all_histories:
                all_histories[name] = []
                all_utilities[name] = []
            all_histories[name].append(p.get_nw_history())
            all_utilities[name].append(
                utility_scorer.score(result.spending[j], config))

    first_key = next(iter(all_histories))
    ages = list(range(config.start_age,
                      config.start_age + len(all_histories[first_key][0])))

    results: Dict[str, Dict] = {}
    for name, histories in all_histories.items():
        arr = np.array(histories) / 1000
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
            'mean_ce': utility_scorer.certainty_equivalent(mean_u, sim_years),
            'median_utility': median_u,
            'median_ce': utility_scorer.certainty_equivalent(median_u, sim_years),
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
        ce = data.get('mean_ce', 0)
        ax.set_title(f'{name}\nE[CE]={format_money(ce)}/yr')
        ax.set_xlabel('Age')
        ax.legend(fontsize=8)

    axes[0].set_ylabel('Net Worth ($M)')
    fig.suptitle('Monte Carlo Retirement Simulation')
    plt.tight_layout()
    plt.show()

    # Print utility summary
    print(f"\n=== Utility Summary ===")
    print(f"{'Portfolio':<28s}  {'E[Utility]':>10s}  {'E[CE]':>12s}  "
          f"{'Med[CE]':>12s}")
    print("-" * 68)
    for name, data in results.items():
        print(f"{name:<28s}  {data['mean_utility']:>10,.0f}  "
              f"{format_money(data['mean_ce']):>12s}/yr  "
              f"{format_money(data['median_ce']):>12s}/yr")
    print()


# ---------------------------------------------------------------------------
# Leverage Sweep
# ---------------------------------------------------------------------------

def run_leverage_sweep(
    config: SimulationConfig,
    spending_rule: SpendingRule | None = None,
    utility_scorer: UtilityScorer | None = None,
    leverage_range: List[float] | None = None,
    n_simulations: int = 500,
) -> Dict[str, List]:
    """Sweep across leverage ratios and compute risk/reward statistics."""
    if spending_rule is None:
        spending_rule = FixedSpending(config.initial_expenses, config.lifestyle_inflation)
    if utility_scorer is None:
        utility_scorer = CRRAUtility.from_config(config)
    if leverage_range is None:
        leverage_range = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5]

    sim_years = config.expected_lifespan - config.start_age + 1

    results: Dict[str, List] = {
        'leverage': [], 'median_nw': [],
        'p10_nw': [], 'p25_nw': [], 'p75_nw': [], 'p90_nw': [],
        'ruin_pct': [], 'mean_ce': [], 'median_ce': [],
    }

    for lev in leverage_range:
        cfg = replace(config, leverage_ratio=lev)
        final_nws = []
        utilities = []
        ruin_count = 0

        for _ in range(n_simulations):
            result = run_simulation(cfg, spending_rule, quiet=True)
            lev_portfolio = result.portfolios.portfolios[2]
            history = lev_portfolio.get_nw_history()
            final_nws.append(history[-1])
            if min(history) < 0:
                ruin_count += 1
            utilities.append(utility_scorer.score(result.spending[2], cfg))

        arr = np.array(final_nws) / 1000
        mean_u = float(np.mean(utilities))
        median_u = float(np.median(utilities))
        mean_ce = utility_scorer.certainty_equivalent(mean_u, sim_years)
        median_ce = utility_scorer.certainty_equivalent(median_u, sim_years)

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
    """Plot leverage sweep: NW percentiles, ruin probability, and CE."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    kelly = compute_optimal_leverage(config)
    levs = sweep['leverage']

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
# Retirement Age Sweep
# ---------------------------------------------------------------------------

def run_retirement_sweep(
    config: SimulationConfig,
    spending_rule: SpendingRule | None = None,
    utility_scorer_factory: Callable[[SimulationConfig], UtilityScorer] | None = None,
    retirement_ages: List[int] | None = None,
    n_simulations: int = 500,
) -> Dict[str, Dict]:
    """Sweep retirement age and find utility-optimal retirement age.

    For each candidate retirement age, runs Monte Carlo with amortized
    spending and computes E[utility] per portfolio strategy.

    Args:
        utility_scorer_factory: Takes a config (with updated retirement_age)
            and returns a UtilityScorer. Default builds CRRAUtility with
            the config's fire_multiplier.
    """
    if spending_rule is None:
        spending_rule = AmortizedSpending()
    if utility_scorer_factory is None:
        def utility_scorer_factory(cfg: SimulationConfig) -> UtilityScorer:
            return CRRAUtility.from_config(cfg)
    if retirement_ages is None:
        retirement_ages = list(range(30, 71, 2))

    sim_years = config.expected_lifespan - config.start_age + 1
    portfolio_results: Dict[str, Dict] = {}

    for ret_age in retirement_ages:
        cfg = replace(config, retirement_age=ret_age)
        scorer = utility_scorer_factory(cfg)

        utilities_by_portfolio: Dict[str, List[float]] = {}

        for _ in range(n_simulations):
            result = run_simulation(cfg, spending_rule, quiet=True)
            for j, p in enumerate(result.portfolios.portfolios):
                name = type(p).__name__
                if name not in utilities_by_portfolio:
                    utilities_by_portfolio[name] = []
                utilities_by_portfolio[name].append(
                    scorer.score(result.spending[j], cfg))

        # Aggregate
        parts = []
        for name, utils in utilities_by_portfolio.items():
            if name not in portfolio_results:
                portfolio_results[name] = {
                    'retirement_ages': [],
                    'mean_utilities': [],
                    'mean_ces': [],
                    'median_ces': [],
                }
            u_arr = np.array(utils)
            mean_u = float(np.mean(u_arr))
            median_u = float(np.median(u_arr))
            portfolio_results[name]['retirement_ages'].append(ret_age)
            portfolio_results[name]['mean_utilities'].append(mean_u)
            portfolio_results[name]['mean_ces'].append(
                scorer.certainty_equivalent(mean_u, sim_years))
            portfolio_results[name]['median_ces'].append(
                scorer.certainty_equivalent(median_u, sim_years))
            ce_str = format_money(portfolio_results[name]['mean_ces'][-1])
            parts.append(f"{name[:15]}: E[CE]={ce_str}/yr")

        print(f"  Retire@{ret_age}: {' | '.join(parts)}")

    # Find optimal age per portfolio
    for name, data in portfolio_results.items():
        best_idx = int(np.argmax(data['mean_ces']))
        data['optimal_age'] = data['retirement_ages'][best_idx]
        data['optimal_ce'] = data['mean_ces'][best_idx]

    return portfolio_results


def plot_retirement_sweep(
    results: Dict[str, Dict],
    config: SimulationConfig,
) -> None:
    """Plot CE spending vs retirement age for each portfolio."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for name, data in results.items():
        ages = data['retirement_ages']
        ces = [c * 1000 for c in data['mean_ces']]
        ax1.plot(ages, ces, 'o-', linewidth=2, label=name, markersize=4)

        opt_age = data['optimal_age']
        opt_ce = data['optimal_ce'] * 1000
        ax1.annotate(f'{opt_age}',
                     xy=(opt_age, opt_ce),
                     xytext=(5, 10), textcoords='offset points',
                     fontsize=9, fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color='red'))

    ax1.set_xlabel('Retirement Age')
    ax1.set_ylabel('E[CE] Spending ($k/yr)')
    vf = config.vitality_floor
    ax1.set_title(f'Optimal Retirement Age\n'
                  f'(FIRE={config.fire_multiplier}, vit floor={vf}, U=c^{config.utility_power})')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    for name, data in results.items():
        ages = data['retirement_ages']
        med_ces = [c * 1000 for c in data['median_ces']]
        ax2.plot(ages, med_ces, 'o-', linewidth=2, label=name, markersize=4)
    ax2.set_xlabel('Retirement Age')
    ax2.set_ylabel('Median CE Spending ($k/yr)')
    ax2.set_title('Median-path CE vs Retirement Age')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Retirement Age Optimization (Amortized Spending)')
    plt.tight_layout()
    plt.show()

    # Summary table
    print(f"\n=== Retirement Age Sweep Summary ===")
    print(f"FIRE={config.fire_multiplier}, "
          f"U(c)=c^{config.utility_power}, "
          f"δ={config.discount_rate:.0%}, "
          f"vitality(peak={config.vitality_peak_age}, "
          f"hl={config.vitality_half_life:.0f}, "
          f"floor={config.vitality_floor})")
    print(f"{'Portfolio':<28s}  {'Optimal Age':>11s}  {'E[CE]':>12s}")
    print("-" * 55)
    for name, data in results.items():
        print(f"{name:<28s}  {data['optimal_age']:>11d}  "
              f"{format_money(data['optimal_ce']):>12s}/yr")
    print()


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
                        help='Run leverage sweep with N sims per level')
    parser.add_argument('--retirement-sweep', type=int, default=0, metavar='N',
                        help='Sweep retirement ages with N sims per age')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    # Decision model
    parser.add_argument('--amortized', action='store_true',
                        help='Use lifecycle amortized spending instead of fixed')
    # Utility model
    parser.add_argument('--discount-rate', type=float, default=0.03,
                        help='Annual time discount rate for utility (default: 0.03)')
    parser.add_argument('--utility-power', type=float, default=0.8,
                        help='CRRA exponent: U(c) = c^α (default: 0.8)')
    parser.add_argument('--fire-multiplier', type=float, default=1.0,
                        help='Utility multiplier for retirement years (default: 1.0)')
    # Vitality curve
    parser.add_argument('--vitality-peak', type=int, default=30,
                        help='Age of peak vitality (default: 30)')
    parser.add_argument('--vitality-half-life', type=float, default=35.0,
                        help='Vitality half-life in years from peak (default: 35)')
    parser.add_argument('--vitality-floor', type=float, default=0.3,
                        help='Minimum vitality multiplier (default: 0.3)')
    parser.add_argument('--no-vitality', action='store_true',
                        help='Disable vitality weighting (floor=1.0)')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

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
        leverage_ratio=args.leverage,
        utility_power=args.utility_power,
        discount_rate=args.discount_rate,
        fire_multiplier=args.fire_multiplier,
        vitality_peak_age=args.vitality_peak,
        vitality_half_life=args.vitality_half_life,
        vitality_floor=1.0 if args.no_vitality else args.vitality_floor,
    )

    if args.optimal_leverage:
        config.leverage_ratio = compute_optimal_leverage(config)
        print(f"Using Kelly-optimal leverage: {config.leverage_ratio:.2f}x")

    # Build spending rule (Decision Model)
    if args.amortized or args.retirement_sweep > 0:
        spending_rule = AmortizedSpending()
    else:
        spending_rule = FixedSpending(config.initial_expenses, config.lifestyle_inflation)

    # Build utility scorer (Utility Model)
    utility_scorer = CRRAUtility.from_config(config)

    sim_years = config.expected_lifespan - config.start_age + 1

    if args.retirement_sweep > 0:
        print(f"Running retirement age sweep "
              f"({args.retirement_sweep} sims per age)...")
        vit_desc = (f"vitality(peak={config.vitality_peak_age}, "
                    f"hl={config.vitality_half_life:.0f}, "
                    f"floor={config.vitality_floor})")
        print(f"FIRE={config.fire_multiplier}, "
              f"U(c)=c^{config.utility_power}, "
              f"δ={config.discount_rate:.0%}, {vit_desc}")
        print()
        sweep_results = run_retirement_sweep(
            config, spending_rule,
            n_simulations=args.retirement_sweep)
        plot_retirement_sweep(sweep_results, config)
    elif args.leverage_sweep > 0:
        print(f"Running leverage sweep ({args.leverage_sweep} sims per level)...")
        kelly = compute_optimal_leverage(config)
        print(f"Kelly optimal: {kelly:.2f}x  |  Half-Kelly: {kelly/2:.2f}x")
        print(f"Spending: {type(spending_rule).__name__}")
        print()
        sweep = run_leverage_sweep(config, spending_rule, utility_scorer,
                                   n_simulations=args.leverage_sweep)
        plot_leverage_sweep(sweep, config)
    elif args.monte_carlo > 0:
        print(f"Running {args.monte_carlo} Monte Carlo simulations...")
        print(f"Spending: {type(spending_rule).__name__}")
        results = run_monte_carlo(config, spending_rule, utility_scorer,
                                  n_simulations=args.monte_carlo)
        plot_monte_carlo(results, config)
    else:
        result = run_simulation(config, spending_rule, seed=args.seed)
        vfloor = config.vitality_floor
        print(f"\n=== Lifetime Utility (U(c) = c^{config.utility_power}, "
              f"δ={config.discount_rate:.0%}, FIRE={config.fire_multiplier}, "
              f"vitality floor={vfloor}) ===")
        print(f"{'Portfolio':<30s}  {'Utility':>10s}  {'CE Spending':>14s}")
        print("-" * 58)
        for j, p in enumerate(result.portfolios.portfolios):
            u = utility_scorer.score(result.spending[j], config)
            ce = utility_scorer.certainty_equivalent(u, sim_years)
            print(f"{type(p).__name__:<30s}  {u:>10,.0f}  {format_money(ce):>14s}/yr")
        print()
        plot_results(result, config)


if __name__ == '__main__':
    main()
