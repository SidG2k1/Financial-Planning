"""Reproducible common-random-number parameter sweeps."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .domain import MarketModelConfig, PlanningScenario, SimulationSettings
from .simulation import MonteCarloEngine, SimulationResult

FloatArray = NDArray[np.float64]
Metric = Literal[
    "mean_utility",
    "median_utility",
    "median_certainty_equivalent",
    "mean_certainty_equivalent",
    "insolvency_probability",
    "policy_shortfall_probability",
    "median_terminal_wealth",
]
_MetricReducer = Callable[[SimulationResult], float]


@dataclass(frozen=True, slots=True)
class SweepResult:
    parameter: str
    values: tuple[float, ...]
    metrics: tuple[float, ...]
    utilities: tuple[float, ...]
    metric: Metric

    @property
    def optimum_index(self) -> int:
        return int(np.argmax(self.utilities))

    @property
    def optimum(self) -> tuple[float, float]:
        index = self.optimum_index
        return self.values[index], self.utilities[index]


@dataclass(frozen=True, slots=True)
class Sweep2DResult:
    x_parameter: str
    y_parameter: str
    x_values: tuple[float, ...]
    y_values: tuple[float, ...]
    metrics: FloatArray
    utilities: FloatArray
    metric: Metric

    @property
    def optimum(self) -> tuple[float, float, float]:
        flat_index = int(np.argmax(self.utilities))
        y_index, x_index = np.unravel_index(flat_index, self.utilities.shape)
        return (
            self.x_values[x_index],
            self.y_values[y_index],
            float(self.utilities[y_index, x_index]),
        )


def _mean_utility(result: SimulationResult) -> float:
    return float(np.mean(result.utility_scores))


def _median_utility(result: SimulationResult) -> float:
    return float(np.median(result.utility_scores))


def _median_certainty_equivalent(result: SimulationResult) -> float:
    return float(np.median(result.certainty_equivalents))


def _mean_certainty_equivalent(result: SimulationResult) -> float:
    return float(np.mean(result.certainty_equivalents))


def _insolvency_probability(result: SimulationResult) -> float:
    return result.insolvency_probability


def _policy_shortfall_probability(result: SimulationResult) -> float:
    return result.policy_shortfall_probability


def _median_terminal_wealth(result: SimulationResult) -> float:
    return float(np.median(result.wealth_paths[:, -1]))


METRIC_REDUCERS: dict[Metric, _MetricReducer] = {
    "mean_utility": _mean_utility,
    "median_utility": _median_utility,
    "median_certainty_equivalent": _median_certainty_equivalent,
    "mean_certainty_equivalent": _mean_certainty_equivalent,
    "insolvency_probability": _insolvency_probability,
    "policy_shortfall_probability": _policy_shortfall_probability,
    "median_terminal_wealth": _median_terminal_wealth,
}
SUPPORTED_METRICS: tuple[Metric, ...] = tuple(METRIC_REDUCERS)


def _metric(result: SimulationResult, metric: Metric) -> float:
    try:
        return METRIC_REDUCERS[metric](result)
    except KeyError as error:
        raise ValueError(f"unknown metric {metric!r}") from error


def _with_parameter(
    scenario: PlanningScenario,
    settings: SimulationSettings,
    market: MarketModelConfig,
    parameter: str,
    value: float,
) -> tuple[PlanningScenario, SimulationSettings, MarketModelConfig]:
    if parameter == "retirement_age":
        person = replace(scenario.person, retirement_age=round(value))
        return replace(scenario, person=person), settings, market
    if parameter == "social_security_claim_age":
        person = replace(scenario.person, social_security_claim_age=round(value))
        return replace(scenario, person=person), settings, market
    if parameter == "annuitization_fraction":
        preferences = replace(scenario.preferences, annuitization_fraction=value)
        return replace(scenario, preferences=preferences), settings, market
    if parameter == "leverage":
        return scenario, replace(settings, leverage=value), market
    if parameter == "risk_tolerance":
        preferences = replace(scenario.preferences, risk_tolerance=value)
        return replace(scenario, preferences=preferences), settings, market
    if parameter == "time_preference":
        preferences = replace(scenario.preferences, time_preference=value)
        return replace(scenario, preferences=preferences), settings, market
    if parameter == "bequest_strength":
        preferences = replace(scenario.preferences, bequest_strength=value)
        return replace(scenario, preferences=preferences), settings, market
    if parameter == "equity_risk_premium":
        return scenario, settings, replace(market, equity_risk_premium=value)
    if parameter == "equity_volatility":
        return scenario, settings, replace(market, equity_volatility=value)
    if parameter == "initial_real_rate":
        return scenario, settings, replace(market, initial_real_rate=value)
    raise ValueError(f"unsupported sweep parameter {parameter!r}")


def _engine_with_market(
    engine: MonteCarloEngine,
    market: MarketModelConfig,
) -> MonteCarloEngine:
    return MonteCarloEngine(
        planner=engine.planner,
        market=engine.market.with_config(market),
        tax_policy=engine.tax_policy,
        leverage_terms=engine.leverage_terms,
        utility_addons=engine.utility_addons,
        rolling_decision_policy=engine.rolling_decision_policy,
    )


def _run_point(
    engine: MonteCarloEngine,
    scenario: PlanningScenario,
    settings: SimulationSettings,
    market: MarketModelConfig,
    metric: Metric,
) -> tuple[float, float]:
    point_engine = _engine_with_market(engine, market)
    result = point_engine.simulate(scenario, settings=settings)
    return _metric(result, metric), _mean_utility(result)


def parameter_sweep(
    engine: MonteCarloEngine,
    scenario: PlanningScenario,
    settings: SimulationSettings,
    *,
    parameter: str,
    values: Sequence[float] | FloatArray,
    metric: Metric = "mean_utility",
) -> SweepResult:
    if len(values) == 0:
        raise ValueError("values cannot be empty")
    output: list[float] = []
    utilities: list[float] = []
    for value in values:
        point_scenario, point_settings, point_market = _with_parameter(
            scenario,
            settings,
            engine.market.config,
            parameter,
            float(value),
        )
        point_metric, point_utility = _run_point(
            engine,
            point_scenario,
            point_settings,
            point_market,
            metric,
        )
        output.append(point_metric)
        utilities.append(point_utility)
    return SweepResult(
        parameter,
        tuple(map(float, values)),
        tuple(output),
        tuple(utilities),
        metric,
    )


def parameter_sweep_2d(
    engine: MonteCarloEngine,
    scenario: PlanningScenario,
    settings: SimulationSettings,
    *,
    x_parameter: str,
    x_values: Sequence[float] | FloatArray,
    y_parameter: str,
    y_values: Sequence[float] | FloatArray,
    metric: Metric = "mean_utility",
) -> Sweep2DResult:
    if len(x_values) == 0 or len(y_values) == 0:
        raise ValueError("sweep axes cannot be empty")
    output = np.zeros((len(y_values), len(x_values)))
    utilities = np.zeros_like(output)
    for y_index, y_value in enumerate(y_values):
        y_scenario, y_settings, y_market = _with_parameter(
            scenario,
            settings,
            engine.market.config,
            y_parameter,
            float(y_value),
        )
        y_engine = _engine_with_market(engine, y_market)
        for x_index, x_value in enumerate(x_values):
            x_scenario, x_settings, x_market = _with_parameter(
                y_scenario,
                y_settings,
                y_engine.market.config,
                x_parameter,
                float(x_value),
            )
            point_metric, point_utility = _run_point(
                y_engine,
                x_scenario,
                x_settings,
                x_market,
                metric,
            )
            output[y_index, x_index] = point_metric
            utilities[y_index, x_index] = point_utility
    return Sweep2DResult(
        x_parameter,
        y_parameter,
        tuple(map(float, x_values)),
        tuple(map(float, y_values)),
        output,
        utilities,
        metric,
    )
