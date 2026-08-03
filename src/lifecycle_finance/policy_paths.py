"""Vectorized accounting for annual policy paths."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .domain import OutcomeType
from .return_models import MarketPaths
from .utility import UtilityModel, UtilityOutcome

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class PolicyPathContext:
    """State visible to a policy before one annual decision."""

    year: int
    age: float
    liquid_wealth: FloatArray
    restricted_equity: FloatArray


@dataclass(frozen=True, slots=True)
class PolicyPathDecision:
    """One year's requested cash flows, allocation, and utility outcomes."""

    consumption: ArrayLike
    target_total_equity: ArrayLike
    external_income: ArrayLike = 0.0
    restricted_vesting: ArrayLike = 0.0
    restricted_release: ArrayLike = 0.0
    event_spending: ArrayLike = 0.0
    event_outcome: OutcomeType | None = None
    outcomes: Mapping[OutcomeType, ArrayLike] = field(default_factory=dict)


class PolicyPathPolicy(Protocol):
    """Policy interface consumed by :class:`PolicyPathEvaluator`."""

    def decide(self, context: PolicyPathContext) -> PolicyPathDecision:
        """Return a decision for the supplied annual state."""


@dataclass(frozen=True, slots=True)
class PolicyPathResult:
    """Candidate-by-path accounting records retained for every return period."""

    beginning_total_wealth: FloatArray
    invested_liquid_wealth: FloatArray
    invested_restricted_equity: FloatArray
    consumption: FloatArray
    event_spending: FloatArray
    target_total_equity: FloatArray
    actual_total_equity: FloatArray
    liquid_equity_weight: FloatArray
    decision_outcomes: Mapping[OutcomeType, FloatArray]
    ordinary_shortfall: BoolArray
    event_shortfall: BoolArray
    insolvent: BoolArray
    utility_scores: FloatArray
    utility_component_scores: Mapping[str, FloatArray]
    preference_breaches: Mapping[str, BoolArray]


def _as_finite_array(value: ArrayLike, *, name: str) -> FloatArray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _align(
    value: ArrayLike,
    *,
    candidate_count: int,
    paths: int,
    name: str,
) -> FloatArray:
    """Align scalar, candidate, path, or candidate-by-path values."""

    array = _as_finite_array(value, name=name)
    shape = (candidate_count, paths)
    if array.ndim == 0:
        return np.full(shape, float(array))
    if array.ndim == 1:
        candidate_aligned = array.shape == (candidate_count,)
        path_aligned = array.shape == (paths,)
        if candidate_aligned and path_aligned:
            raise ValueError(
                f"{name} is ambiguous when candidate_count equals paths; "
                "use an explicit row or column vector"
            )
        if candidate_aligned:
            return np.broadcast_to(array[:, np.newaxis], shape).copy()
        if path_aligned:
            return np.broadcast_to(array[np.newaxis, :], shape).copy()
    if array.shape == (candidate_count, 1):
        return np.broadcast_to(array, shape).copy()
    if array.shape == (1, paths):
        return np.broadcast_to(array, shape).copy()
    if array.shape == shape:
        return array.copy()
    raise ValueError(
        f"{name} must be scalar, a candidate or path vector, or have shape {shape}"
    )


def _validate_market(market: MarketPaths) -> tuple[FloatArray, FloatArray, FloatArray]:
    returns = (
        ("equity_returns", market.equity_returns),
        ("bond_returns", market.bond_returns),
        ("cash_returns", market.cash_returns),
    )
    validated: list[FloatArray] = []
    shape: tuple[int, int] | None = None
    for name, values in returns:
        array = _as_finite_array(values, name=name)
        if array.ndim != 2:
            raise ValueError(f"{name} must be a path-by-year array")
        if shape is None:
            shape = array.shape
            if shape[0] == 0 or shape[1] == 0:
                raise ValueError("market returns must contain at least one path and year")
        elif array.shape != shape:
            raise ValueError(f"{name} must align with equity_returns")
        if np.any(array <= -1.0):
            raise ValueError(f"{name} must be greater than -1")
        validated.append(array)
    real_rates = _as_finite_array(market.real_rates, name="real_rates")
    assert shape is not None
    if real_rates.shape != shape:
        raise ValueError("real_rates must align with equity_returns")
    return tuple(validated)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class PolicyPathEvaluator:
    """Evaluate all candidates and market paths with a single annual loop."""

    utility_model: UtilityModel | None = None

    def evaluate(
        self,
        market: MarketPaths,
        policy: PolicyPathPolicy,
        *,
        candidate_count: int,
        ages: Sequence[float],
        exposure: ArrayLike,
        initial_liquid_wealth: ArrayLike,
        initial_restricted_equity: ArrayLike = 0.0,
    ) -> PolicyPathResult:
        equity_returns, bond_returns, _ = _validate_market(market)
        if candidate_count <= 0:
            raise ValueError("candidate_count must be positive")
        paths, horizon = equity_returns.shape
        age_values = _as_finite_array(ages, name="ages")
        if age_values.shape != (horizon + 1,):
            raise ValueError("ages must provide one boundary age per return period")
        exposure_values = _as_finite_array(exposure, name="exposure")
        if exposure_values.ndim > 1 or exposure_values.size not in (1, horizon + 1):
            raise ValueError("exposure must be scalar or align with ages")
        if np.any((exposure_values < 0.0) | (exposure_values > 1.0)):
            raise ValueError("exposure must be between 0 and 1")

        liquid_wealth = _align(
            initial_liquid_wealth,
            candidate_count=candidate_count,
            paths=paths,
            name="initial_liquid_wealth",
        )
        restricted_equity = _align(
            initial_restricted_equity,
            candidate_count=candidate_count,
            paths=paths,
            name="initial_restricted_equity",
        )
        if np.any(liquid_wealth < 0.0) or np.any(restricted_equity < 0.0):
            raise ValueError("initial wealth must be nonnegative")

        shape = (candidate_count, paths, horizon)
        beginning_total_wealth = np.empty(shape)
        invested_liquid_wealth = np.empty(shape)
        invested_restricted_equity = np.empty(shape)
        consumption_history = np.empty(shape)
        event_spending_history = np.zeros(shape)
        target_total_equity_history = np.empty(shape)
        actual_total_equity = np.empty(shape)
        liquid_equity_weight = np.empty(shape)
        ordinary_shortfall = np.zeros(shape, dtype=bool)
        event_shortfall = np.zeros(shape, dtype=bool)
        insolvent = np.zeros(shape, dtype=bool)
        decision_outcomes: dict[OutcomeType, FloatArray] = {}

        for year in range(horizon):
            beginning_total_wealth[:, :, year] = liquid_wealth + restricted_equity
            decision = policy.decide(
                PolicyPathContext(
                    year=year,
                    age=float(age_values[year]),
                    liquid_wealth=liquid_wealth.copy(),
                    restricted_equity=restricted_equity.copy(),
                )
            )
            consumption = _align(
                decision.consumption,
                candidate_count=candidate_count,
                paths=paths,
                name="consumption",
            )
            target_total_equity = _align(
                decision.target_total_equity,
                candidate_count=candidate_count,
                paths=paths,
                name="target_total_equity",
            )
            external_income = _align(
                decision.external_income,
                candidate_count=candidate_count,
                paths=paths,
                name="external_income",
            )
            restricted_vesting = _align(
                decision.restricted_vesting,
                candidate_count=candidate_count,
                paths=paths,
                name="restricted_vesting",
            )
            restricted_release = _align(
                decision.restricted_release,
                candidate_count=candidate_count,
                paths=paths,
                name="restricted_release",
            )
            event_spending = _align(
                decision.event_spending,
                candidate_count=candidate_count,
                paths=paths,
                name="event_spending",
            )
            for name, values in (
                ("consumption", consumption),
                ("external_income", external_income),
                ("restricted_vesting", restricted_vesting),
                ("restricted_release", restricted_release),
                ("event_spending", event_spending),
            ):
                if np.any(values < 0.0):
                    raise ValueError(f"{name} must be nonnegative")
            if np.any(restricted_release > restricted_equity):
                raise ValueError("restricted_release cannot exceed restricted_equity")
            if np.any((target_total_equity < 0.0) | (target_total_equity > 1.0)):
                raise ValueError("target_total_equity must be between 0 and 1")

            aligned_outcomes = {
                OutcomeType(outcome): _align(
                    value,
                    candidate_count=candidate_count,
                    paths=paths,
                    name=f"outcomes[{outcome}]",
                )
                for outcome, value in decision.outcomes.items()
            }
            if decision.event_outcome is not None:
                event_outcome = OutcomeType(decision.event_outcome)
                if event_outcome in aligned_outcomes:
                    raise ValueError("event_outcome cannot duplicate outcomes")
            elif np.any(event_spending != 0.0):
                raise ValueError("event_outcome is required when event_spending is nonzero")

            liquid_wealth = liquid_wealth + restricted_release
            restricted_equity = restricted_equity - restricted_release
            liquid_wealth = liquid_wealth + external_income
            restricted_equity = restricted_equity + restricted_vesting
            ordinary_shortfall[:, :, year] = consumption > liquid_wealth
            funded_consumption = np.minimum(consumption, liquid_wealth)
            liquid_wealth = liquid_wealth - funded_consumption
            event_shortfall[:, :, year] = event_spending > liquid_wealth
            funded_event_spending = np.minimum(event_spending, liquid_wealth)
            liquid_wealth = liquid_wealth - funded_event_spending
            insolvent[:, :, year] = (
                ordinary_shortfall[:, :, year]
                & (liquid_wealth == 0.0)
                & (restricted_equity == 0.0)
            )
            if decision.event_outcome is not None:
                aligned_outcomes[event_outcome] = funded_event_spending
            for outcome, values in aligned_outcomes.items():
                decision_outcomes.setdefault(outcome, np.zeros(shape))[:, :, year] = values

            total_wealth = liquid_wealth + restricted_equity
            liquid_equity = np.clip(
                target_total_equity * total_wealth - restricted_equity,
                0.0,
                liquid_wealth,
            )
            invested_liquid_wealth[:, :, year] = liquid_wealth
            invested_restricted_equity[:, :, year] = restricted_equity
            consumption_history[:, :, year] = funded_consumption
            event_spending_history[:, :, year] = funded_event_spending
            target_total_equity_history[:, :, year] = target_total_equity
            with np.errstate(invalid="ignore", divide="ignore"):
                liquid_equity_weight[:, :, year] = np.divide(
                    liquid_equity,
                    liquid_wealth,
                    out=np.zeros_like(liquid_wealth),
                    where=liquid_wealth > 0.0,
                )
                actual_total_equity[:, :, year] = np.divide(
                    restricted_equity + liquid_equity,
                    total_wealth,
                    out=np.zeros_like(total_wealth),
                    where=total_wealth > 0.0,
                )

            liquid_wealth = liquid_equity * (1.0 + equity_returns[:, year]) + (
                liquid_wealth - liquid_equity
            ) * (1.0 + bond_returns[:, year])
            restricted_equity = restricted_equity * (1.0 + equity_returns[:, year])

        utility_scores, utility_component_scores, preference_breaches = (
            self._score_utility(
                consumption=consumption_history,
                decision_outcomes=decision_outcomes,
                terminal_wealth=liquid_wealth + restricted_equity,
                ages=age_values[:-1],
                exposure=exposure_values,
            )
        )

        return PolicyPathResult(
            beginning_total_wealth=beginning_total_wealth,
            invested_liquid_wealth=invested_liquid_wealth,
            invested_restricted_equity=invested_restricted_equity,
            consumption=consumption_history,
            event_spending=event_spending_history,
            target_total_equity=target_total_equity_history,
            actual_total_equity=actual_total_equity,
            liquid_equity_weight=liquid_equity_weight,
            decision_outcomes=decision_outcomes,
            ordinary_shortfall=ordinary_shortfall,
            event_shortfall=event_shortfall,
            insolvent=insolvent,
            utility_scores=utility_scores,
            utility_component_scores=utility_component_scores,
            preference_breaches=preference_breaches,
        )

    def _score_utility(
        self,
        *,
        consumption: FloatArray,
        decision_outcomes: Mapping[OutcomeType, FloatArray],
        terminal_wealth: FloatArray,
        ages: FloatArray,
        exposure: FloatArray,
    ) -> tuple[FloatArray, Mapping[str, FloatArray], Mapping[str, BoolArray]]:
        candidate_count, paths, horizon = consumption.shape
        score_shape = (candidate_count, paths)
        if self.utility_model is None:
            return np.zeros(score_shape), {}, {}

        if exposure.size == 1:
            annual_exposure = np.full(horizon, float(exposure[0]))
        else:
            annual_exposure = exposure[:horizon]
        aligned_exposure = np.broadcast_to(annual_exposure, consumption.shape)
        utility_outcome = UtilityOutcome(
            spending=consumption.reshape(candidate_count * paths, horizon),
            exposure=aligned_exposure.reshape(candidate_count * paths, horizon),
            ages=tuple(float(age) for age in ages),
            terminal_wealth=terminal_wealth.reshape(candidate_count * paths),
            decisions={
                outcome: values.reshape(candidate_count * paths, horizon)
                for outcome, values in decision_outcomes.items()
            },
        )
        active_addons = self.utility_model.validate_outcomes(
            utility_outcome.available_outcomes
        )
        components = self.utility_model.decompose(utility_outcome)
        component_scores = {
            name: values.reshape(score_shape) for name, values in components.items()
        }
        if component_scores:
            scores = np.sum(np.stack(tuple(component_scores.values())), axis=0)
        else:
            scores = np.zeros(score_shape)

        preference_breaches: dict[str, BoolArray] = {}
        age_values = np.asarray(utility_outcome.ages)
        for addon in active_addons:
            diagnostic_breach = getattr(addon.curve, "diagnostic_breach", None)
            values = utility_outcome.values(addon.outcome)
            if not callable(diagnostic_breach) or values.shape != utility_outcome.spending.shape:
                continue
            age_mask = np.ones(horizon, dtype=bool)
            if addon.minimum_age is not None:
                age_mask &= age_values >= addon.minimum_age
            if addon.maximum_age is not None:
                age_mask &= age_values <= addon.maximum_age
            preference_breaches[addon.name] = (
                np.asarray(diagnostic_breach(values), dtype=bool)
                & age_mask[np.newaxis, :]
                & (utility_outcome.exposure > 0.0)
            ).reshape(consumption.shape)

        return np.asarray(scores, dtype=np.float64), component_scores, preference_breaches
