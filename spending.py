# Decision Model: Spending Rules
# All monetary values are in $1,000 units (2023 USD)

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

from config import SimulationConfig, post_tax
from models import vitality_at_age


@dataclass
class YearContext:
    """All info a spending rule needs for one year's decision."""
    nw: float               # portfolio NW after returns
    income: float           # post-tax income this year ($1k), includes SS if applicable
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
    ss_benefit: float = 0.0 # annual post-tax SS benefit ($1k), for PV calc


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


def _planning_remaining(ctx: YearContext) -> int:
    """Remaining years for spending planning.

    Normally plans to expected_lifespan. If the person outlives that
    (stochastic lifespan), extends the horizon to max_age so spending
    doesn't cliff to zero.
    """
    remaining = ctx.expected_lifespan - ctx.age + 1
    if remaining <= 0:
        remaining = ctx.config.max_age - ctx.age + 1
    return max(remaining, 1)


def _compute_total_resources(ctx: YearContext) -> Tuple[float, float]:
    """Compute total lifetime resources and discount factor for spending rules.

    Returns (total_resources, d) where:
        total = NW * tax_advantage + income + PV(future_income) + PV(SS)
                - PV(future_one_time) - one_time_this_year

    The retirement_tax_advantage inflates the real purchasing power of
    portfolio NW (retirement withdrawals face ~15% LTCG vs ~38% earned income).
    """
    remaining = _planning_remaining(ctx)
    r = max(ctx.bond_yield + ctx.equity_risk_premium, 0.001)
    d = 1.0 / (1.0 + r)

    # PV of future earned income (next year onward until retirement)
    working_years_left = max(0, ctx.retirement_age - ctx.age)
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

    # PV of future SS income
    pv_ss = 0.0
    planning_end = ctx.age + remaining - 1
    if ctx.ss_benefit > 0:
        ss_start = ctx.config.ss_claiming_age
        for future_age in range(max(ss_start, ctx.age + 1),
                                planning_end + 1):
            pv_ss += ctx.ss_benefit * (d ** (future_age - ctx.age))

    # PV of future one-time expenses (next year onward)
    pv_onetime = 0.0
    for future_age, amount in ctx.config.one_time_expenses.items():
        if future_age > ctx.age:
            pv_onetime += amount * (d ** (future_age - ctx.age))

    # Retirement tax advantage: portfolio NW buys more in retirement because
    # withdrawals (LTCG/Roth) are taxed at ~15% vs earned income at ~38%.
    # Only apply after retirement when spending comes from portfolio.
    is_retired = ctx.age >= ctx.retirement_age
    tax_adv = ctx.config.retirement_tax_advantage if is_retired else 1.0

    total = (ctx.nw * tax_adv + ctx.income + pv_income + pv_ss
             - pv_onetime - ctx.one_time_expense)

    return total, d


class AmortizedSpending(SpendingRule):
    """Lifecycle spending: annuitize total resources over remaining lifespan.

    Each year recomputes sustainable spending based on:
        total_resources = NW + income + PV(future_income) + PV(SS) - PV(one_time)
        regular_spending = total_resources / annuity_factor(remaining, r)
    where r = bond_yield + ERP (expected real return, adapts to rate env).

    If spending_floor > 0, resources for the floor are reserved first and
    only the excess is annuitized on top.
    """

    def compute(self, ctx: YearContext) -> float:
        remaining = _planning_remaining(ctx)

        total, d = _compute_total_resources(ctx)

        # Annuity factor: PV of $1/year for `remaining` years at rate r
        annuity = (1.0 - d ** remaining) / (1.0 - d)

        floor = ctx.config.spending_floor
        if floor > 0:
            regular = max(floor, total / annuity)
        else:
            regular = max(0.0, total / annuity)
        return regular + ctx.one_time_expense


class VitalityAmortizedSpending(SpendingRule):
    """Lifecycle spending weighted by vitality -- spend more in high-vitality years.

    Like AmortizedSpending, but replaces the flat annuity with a vitality-weighted
    annuity: spending(age) = total_resources * v(age) / sum(v(age+s) * d^s).
    This front-loads consumption into years when you can enjoy it most.

    If spending_floor > 0, resources for the floor are reserved first
    (PV of floor over remaining years), then the excess is front-loaded
    via vitality weighting on top of the floor.
    """

    def compute(self, ctx: YearContext) -> float:
        remaining = _planning_remaining(ctx)

        total, d = _compute_total_resources(ctx)

        floor = ctx.config.spending_floor

        # Vitality-weighted annuity: sum of v(age+s) * d^s for s=0..remaining-1
        v_annuity = sum(
            vitality_at_age(ctx.age + s, ctx.config) * (d ** s)
            for s in range(remaining)
        )

        if v_annuity <= 0:
            return max(floor, 0.0) + ctx.one_time_expense

        v_now = vitality_at_age(ctx.age, ctx.config)

        if floor > 0:
            # Reserve resources for the floor, front-load only the excess
            flat_annuity = (1.0 - d ** remaining) / (1.0 - d)
            pv_floor = floor * flat_annuity
            excess = max(0.0, total - pv_floor)
            regular = floor + excess * v_now / v_annuity
        else:
            regular = max(0.0, total * v_now / v_annuity)

        return regular + ctx.one_time_expense
