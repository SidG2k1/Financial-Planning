# Financial Model: Simulation Engine
# All monetary values are in $1,000 units (2023 USD)

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from config import SimulationConfig, SimulationResult, format_money, post_tax
from models import (
    compute_optimal_leverage,
    compute_ss_benefit,
    evolve_bond_yield,
    sample_death_age,
)
from portfolio import (
    CashPortfolio,
    LeveragedStockPortfolio,
    PortfolioManager,
    StockPortfolio,
)
from spending import FixedSpending, SpendingRule, YearContext


@dataclass
class YearShocks:
    """Correlated market shocks for one year."""
    bond: float
    stock: float
    vol: float


def _generate_year_shocks(
    rng: np.random.Generator,
    config: SimulationConfig,
    negate: bool,
    use_fat_tails: bool,
    t_scale: float,
    rho: float,
    rho_complement: float,
) -> YearShocks:
    """Generate one year's correlated market shocks."""
    bond = rng.standard_normal()
    indep = rng.standard_t(config.stock_tail_df) * t_scale if use_fat_tails else rng.standard_normal()
    vol = rng.standard_normal()
    if negate:
        bond, indep, vol = -bond, -indep, -vol
    stock = rho * bond + rho_complement * indep
    return YearShocks(bond=bond, stock=stock, vol=vol)


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
    negate_shocks: bool = False,
) -> SimulationResult:
    """Run the retirement simulation with pluggable spending rule.

    Market model:
        bond_yield follows Vasicek mean-reverting process
        stock_return = bond_yield + ERP + sigma_t * eps_stock
        margin_fee   = bond_yield + spread

    Realistic dynamics:
        - Fat tails: eps ~ Student-t(df) normalized to unit variance
        - Stochastic vol: log(sigma_t/sigma_bar) = rho*log(sigma_{t-1}/sigma_bar) + eta*eps_vol
        - Stock-bond correlation: eps_stock = corr*eps_bond + sqrt(1-corr^2)*eps_indep

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

    # SS benefit (computed early for display; depends on retirement_age)
    ss_benefit = compute_ss_benefit(config)

    if not quiet:
        kelly = compute_optimal_leverage(config)
        print(f"Simulation seed: {seed}")
        print(f"E[stock return] = bond_yield + ERP = ~{config.initial_bond_yield + config.equity_risk_premium:.1%}")
        print(f"Margin fee = bond_yield + spread = ~{config.initial_bond_yield + config.margin_spread:.1%}")
        print(f"Kelly optimal leverage: {kelly:.2f}x  |  Half-Kelly: {kelly/2:.2f}x  |  Using: {config.leverage_ratio:.2f}x")
        tail_desc = f"Student-t(df={config.stock_tail_df:.0f})" if config.stock_tail_df <= 100 else "Normal"
        print(f"Return dist: {tail_desc}  |  Vol clustering: rho={config.vol_persistence}, eta={config.vol_of_vol}")
        print(f"Stock-bond corr: {config.stock_bond_corr:+.2f}")
        print(f"Spending: {type(spending_rule).__name__}")
        if config.ss_enabled:
            print(f"SS benefit: {format_money(ss_benefit)}/yr "
                  f"(claiming@{config.ss_claiming_age}, "
                  f"{config.retirement_age - config.start_age} working yrs)")
        print(f"Retirement tax advantage: {config.retirement_tax_advantage:.0%}")
        print()

    # Determine actual lifespan
    if config.stochastic_lifespan:
        death_age = sample_death_age(config, rng)
    else:
        death_age = config.expected_lifespan

    sim_years = config.expected_lifespan - config.start_age + 1
    realized_income, extended_schedule = _build_income_schedule(config)

    # Extend arrays if living beyond expected lifespan
    actual_years = death_age - config.start_age + 1
    if actual_years > sim_years:
        realized_income += [0.0] * (actual_years - sim_years)
        extended_schedule += [0.0] * (actual_years - sim_years)

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
            maintenance_margin=config.maintenance_margin,
            margin_call_leverage=config.margin_call_leverage,
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

    for i, curr_age in enumerate(range(config.start_age, death_age + 1)):
        # --- Generate correlated shocks ---
        shocks = _generate_year_shocks(
            rng, config, negate_shocks, use_fat_tails, t_scale, rho, rho_complement)

        # --- Evolve stochastic volatility ---
        log_vol = config.vol_persistence * log_vol + config.vol_of_vol * shocks.vol
        current_vol = config.stock_vol * np.exp(log_vol)

        # --- Evolve bond yield (Vasicek) ---
        bond_yield = evolve_bond_yield(bond_yield, config, shocks.bond)

        # --- Compute returns ---
        year_return[0] = 1 + bond_yield + config.equity_risk_premium + current_vol * shocks.stock
        year_margin_fee[0] = max(bond_yield + config.margin_spread, 0.0)

        # Stochastic income: market-correlated job loss
        if config.stochastic_income and curr_age < config.retirement_age:
            stock_excess = (year_return[0] - 1) - (bond_yield + config.equity_risk_premium)
            loss_prob = min(
                config.job_loss_base_prob * np.exp(
                    config.job_loss_market_sensitivity * max(0, -stock_excess)),
                0.5)
            if rng.uniform() < loss_prob:
                realized_income[i] *= config.job_loss_income_fraction

        inc = post_tax(realized_income[i], config)
        # Add SS income once past claiming age
        if curr_age >= config.ss_claiming_age and ss_benefit > 0:
            inc += ss_benefit
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
                ss_benefit=ss_benefit,
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

    return SimulationResult(
        portfolios=portfolios, spending=spending_records,
        death_age=death_age if config.stochastic_lifespan else None)
