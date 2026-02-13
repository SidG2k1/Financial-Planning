# Financial models: vitality, Social Security, bond yields, mortality, Bayesian sampling
# All monetary values are in $1,000 units (2023 USD)

from __future__ import annotations

from dataclasses import replace
from typing import List

import numpy as np

from config import SimulationConfig, post_tax


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


def compute_ss_benefit(config: SimulationConfig,
                       retirement_age: int | None = None) -> float:
    """Compute annual post-tax SS benefit ($1k units) given work history.

    Uses 2024 SS bend points. FIRE reduces SS because fewer working years
    means lower AIME (average of top 35 years, zeros fill missing years).

    Args:
        retirement_age: Override for config.retirement_age (used in sweeps).

    Returns:
        Annual post-tax SS benefit in $1k units. 0.0 if ss_enabled is False.
    """
    if not config.ss_enabled:
        return 0.0

    ret_age = retirement_age if retirement_age is not None else config.retirement_age
    working_years = max(0, ret_age - config.start_age)

    # Collect capped earnings for each working year
    schedule = config.income_schedule
    earnings: List[float] = []
    for i in range(working_years):
        if i < len(schedule):
            gross = schedule[i]
        elif schedule:
            gross = schedule[-1]
        else:
            gross = 0.0
        earnings.append(min(gross, config.ss_taxable_max))

    # Top 35 years (zeros fill if fewer than 35 working years)
    earnings.sort(reverse=True)
    top_35 = (earnings[:35] + [0.0] * 35)[:35]

    # AIME: Average Indexed Monthly Earnings ($1k/month)
    aime = sum(top_35) / (35 * 12)

    # PIA bend points (2024, in $1k/month): $1.174k and $7.078k
    b1, b2 = 1.174, 7.078
    if aime <= b1:
        pia = 0.90 * aime
    elif aime <= b2:
        pia = 0.90 * b1 + 0.32 * (aime - b1)
    else:
        pia = 0.90 * b1 + 0.32 * (b2 - b1) + 0.15 * (aime - b2)

    annual_pia = pia * 12  # $1k/year

    # Adjust for claiming age vs FRA
    claim = config.ss_claiming_age
    fra = config.ss_fra
    if claim < fra:
        months_early = (fra - claim) * 12
        # 5/9 of 1% per month for first 36 months, then 5/12 of 1%
        if months_early <= 36:
            reduction = months_early * (5.0 / 9.0) / 100.0
        else:
            reduction = (36 * (5.0 / 9.0) + (months_early - 36) * (5.0 / 12.0)) / 100.0
        annual_pia *= (1.0 - reduction)
    elif claim > fra:
        # Delayed retirement credits: 8% per year past FRA
        annual_pia *= (1.0 + 0.08 * (claim - fra))

    # Tax: 85% of SS is taxable for high earners; apply progressive tax to that
    # For FIRE retirees with low other income, effective rate is modest.
    taxable_portion = 0.85 * annual_pia
    tax = annual_pia - post_tax(taxable_portion, config) if taxable_portion > 0 else 0.0
    # But only 85% was taxable, so actual tax = tax on the taxable portion
    net_ss = annual_pia - tax

    return max(0.0, net_ss)



def evolve_bond_yield(
    current_yield: float,
    config: SimulationConfig,
    bond_shock: float,
) -> float:
    """Advance real bond yield by one year. Floor at -2%."""
    drift = config.bond_yield_mean_reversion * (config.long_run_bond_yield - current_yield)
    new_yield = current_yield + drift + config.bond_yield_vol * bond_shock
    return max(new_yield, -0.02)


def sample_death_age(config: SimulationConfig, rng: np.random.Generator) -> int:
    """Sample a death age from Gompertz mortality, conditioned on survival to start_age.

    Gompertz hazard: h(age) = a * exp(b * age)
    Survival: S(t) = exp(a/b * (1 - exp(b*t)))
    """
    a, b = config.gompertz_a, config.gompertz_b
    s_start = np.exp(a / b * (1 - np.exp(b * config.start_age)))
    u = rng.uniform()
    inner = 1 - (b / a) * np.log(u * s_start)
    if inner <= 0:
        return config.max_age
    death_age = int((1 / b) * np.log(inner))
    return min(max(death_age, config.start_age + 1), config.max_age)


def sample_bayesian_config(
    config: SimulationConfig, rng: np.random.Generator,
) -> SimulationConfig:
    """Sample market parameters from posteriors for one MC simulation."""
    erp = max(0.0, rng.normal(config.equity_risk_premium, config.bayesian_erp_std))
    vol = config.stock_vol * np.exp(rng.normal(0, config.bayesian_vol_std))
    bond = rng.normal(config.long_run_bond_yield, config.bayesian_bond_yield_std)
    return replace(config,
                   equity_risk_premium=erp,
                   stock_vol=vol,
                   initial_bond_yield=bond,
                   long_run_bond_yield=bond)
