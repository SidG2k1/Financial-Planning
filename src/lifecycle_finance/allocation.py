"""Human-capital-aware financial allocation and glide-path helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .domain import Allocation, EconomicExposure


def economic_buckets(
    value: float,
    exposure: EconomicExposure,
    *,
    cash_component: float = 0.0,
) -> Allocation:
    equity = value * exposure.equity_fraction
    global_equity = equity * exposure.global_fraction_of_equity
    return Allocation(
        domestic_equity=equity - global_equity,
        global_equity=global_equity,
        bonds=value - equity,
        cash=cash_component,
    )


def desired_net_worth_buckets(
    net_worth: float,
    risk_tolerance: float,
    global_equity_fraction: float,
) -> Allocation:
    equity = risk_tolerance * net_worth
    global_equity = global_equity_fraction * equity
    return Allocation(
        domestic_equity=equity - global_equity,
        global_equity=global_equity,
        bonds=0.0,
        cash=net_worth - equity,
    )


def financial_allocation(
    *,
    financial_wealth: float,
    desired: Allocation,
    human_capital_risky: float,
    human_capital_cash: float,
    human_capital_exposure: EconomicExposure,
    liability_risky: float,
    liability_cash: float,
    liability_exposure: EconomicExposure,
) -> Allocation:
    """Solve financial buckets = desired net-worth buckets - HC + liabilities."""
    if financial_wealth <= 0:
        return Allocation(0.0, 0.0, 0.0, 1.0)
    human = economic_buckets(
        human_capital_risky,
        human_capital_exposure,
        cash_component=human_capital_cash,
    )
    liability = economic_buckets(
        liability_risky,
        liability_exposure,
        cash_component=liability_cash,
    )
    return Allocation(
        float(
            (desired.domestic_equity - human.domestic_equity + liability.domestic_equity)
            / financial_wealth
        ),
        float(
            (desired.global_equity - human.global_equity + liability.global_equity)
            / financial_wealth
        ),
        float((desired.bonds - human.bonds + liability.bonds) / financial_wealth),
        float((desired.cash - human.cash + liability.cash) / financial_wealth),
    )


def constrain_long_only(allocation: Allocation) -> Allocation:
    """Workbook constraint: zero negative buckets and rescale positive buckets."""
    values = np.array(
        [
            allocation.domestic_equity,
            allocation.global_equity,
            allocation.bonds,
            allocation.cash,
        ],
        dtype=float,
    )
    target_total = values.sum()
    positive = np.maximum(values, 0.0)
    if positive.sum() <= 0 or target_total <= 0:
        return Allocation(0.0, 0.0, 0.0, 1.0)
    constrained = positive * target_total / positive.sum()
    return Allocation(*map(float, constrained))


@dataclass(frozen=True, slots=True)
class AllocationConstraint:
    """General bounded-simplex projection for application-specific constraints."""

    lower: Allocation = field(default_factory=lambda: Allocation(0.0, 0.0, 0.0, 0.0))
    upper: Allocation = field(default_factory=lambda: Allocation(1.0, 1.0, 1.0, 1.0))

    def apply(self, allocation: Allocation) -> Allocation:
        values = np.array(list(allocation.as_dict().values()), dtype=float)
        lower = np.array(list(self.lower.as_dict().values()), dtype=float)
        upper = np.array(list(self.upper.as_dict().values()), dtype=float)
        if np.any(lower > upper) or lower.sum() > 1 or upper.sum() < 1:
            raise ValueError("allocation bounds do not contain the unit simplex")

        # Bisection projects onto {x: sum(x)=1, lower<=x<=upper}.
        low_lambda = float(np.min(values - upper))
        high_lambda = float(np.max(values - lower))
        for _ in range(100):
            midpoint = 0.5 * (low_lambda + high_lambda)
            projected = np.clip(values - midpoint, lower, upper)
            if projected.sum() > 1:
                low_lambda = midpoint
            else:
                high_lambda = midpoint
        projected = np.clip(values - high_lambda, lower, upper)
        projected /= projected.sum()
        return Allocation(*map(float, projected))
