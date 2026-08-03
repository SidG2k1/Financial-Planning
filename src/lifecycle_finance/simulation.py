"""Vectorized stochastic market and retirement simulation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal, overload

import numpy as np
from numpy.typing import NDArray

from .decisions import (
    JointRollingDecisionOptimizer,
    RollingDecisionContext,
    RollingDecisionPolicy,
)
from .demographics import GompertzMortality
from .domain import (
    Allocation,
    LeverageInstrument,
    LifecyclePlan,
    MarketModelConfig,
    OutcomeType,
    PlanningScenario,
    SimulationSettings,
    UtilityAggregation,
)
from .income_risk import (
    IncomeRiskContext,
    IncomeRiskModel,
    IncomeRiskState,
    TransitoryMarketJobLoss,
    _validate_income_risk_state_paths,
    _validate_income_risk_step_paths,
)
from .lifecycle import LifecyclePlanner
from .return_models import (
    MarketPathModel,
    MarketPaths,
    RegimeSwitchingMarket,
)
from .return_models import (
    StochasticMarket as StochasticMarket,
)
from .spending import SpendingContext, SpendingPolicy
from .taxes import TaxPolicy
from .utility import (
    IsoelasticCurve,
    LinearCurve,
    SpendingFloorCurve,
    TargetCurve,
    UtilityAddon,
    UtilityModel,
    UtilityOutcome,
    _weighted_certainty_equivalent,
    vitality,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class LeverageTerms:
    futures_tax_rate: float = 0.268
    futures_financing_spread: float = 0.002
    futures_roll_cost: float = 0.001
    box_financing_spread: float = 0.004
    box_dividend_yield: float = 0.013
    box_dividend_tax_rate: float = 0.238
    box_crisis_spread_widening: float = 0.01
    box_crisis_threshold: float = -0.10


def leveraged_portfolio_return(
    market: MarketPaths,
    year: int,
    allocation: Allocation | FloatArray,
    effective_leverage: FloatArray,
    instrument: LeverageInstrument,
    *,
    market_config: MarketModelConfig,
    leverage_terms: LeverageTerms | None = None,
) -> FloatArray:
    """Apply the engine's financing and tax conventions to one year of returns."""

    terms = LeverageTerms() if leverage_terms is None else leverage_terms
    instrument = LeverageInstrument(instrument)
    if isinstance(allocation, Allocation):
        equity_weight: float | FloatArray = allocation.equity
        bond_weight: float | FloatArray = allocation.bonds
        cash_weight: float | FloatArray = allocation.cash
    else:
        equity_weight = np.asarray(allocation, dtype=float)
        bond_weight = 1.0 - equity_weight
        cash_weight = np.zeros_like(equity_weight)
    leverage = np.asarray(effective_leverage, dtype=float)
    if not np.all(np.isfinite(leverage)) or np.any(leverage < 1.0):
        raise ValueError("effective_leverage must be finite and at least one")

    base = (
        equity_weight * market.equity_returns[:, year]
        + bond_weight * market.bond_returns[:, year]
        + cash_weight * market.cash_returns[:, year]
    )
    overlay = np.maximum(leverage - 1.0, 0.0)
    rates = market.real_rates[:, year]
    excess = market.equity_returns[:, year] - rates
    if instrument is LeverageInstrument.FUTURES:
        financing = rates + terms.futures_financing_spread
        overlay_return = overlay * (
            market.equity_returns[:, year] - financing - terms.futures_roll_cost
        )
        overlay_return -= np.maximum(overlay_return, 0.0) * terms.futures_tax_rate
    elif instrument is LeverageInstrument.BOX_SPREAD:
        crisis = np.where(
            excess < terms.box_crisis_threshold,
            terms.box_crisis_spread_widening,
            0.0,
        )
        financing = rates + terms.box_financing_spread + crisis
        dividend_drag = overlay * terms.box_dividend_yield * terms.box_dividend_tax_rate
        overlay_return = overlay * (market.equity_returns[:, year] - financing)
        overlay_return -= dividend_drag
    else:
        financing = rates + market_config.margin_spread
        overlay_return = overlay * (market.equity_returns[:, year] - financing)
    return np.asarray(base + overlay_return, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class PreferenceDiagnostic:
    """Per-path diagnostic for a utility preference, never a constraint."""

    breach_count: IntArray
    utility_loss: FloatArray

    @property
    def breach_probability(self) -> float:
        return float(np.mean(self.breach_count > 0))

    @property
    def mean_breach_count(self) -> float:
        return float(np.mean(self.breach_count))


@dataclass(frozen=True, slots=True)
class SimulationDiagnostics:
    """Physical and policy diagnostics kept separate from preferences."""

    insolvent: BoolArray
    policy_shortfall: BoolArray
    preferences: Mapping[str, PreferenceDiagnostic]

    @property
    def insolvency_probability(self) -> float:
        return float(np.mean(self.insolvent))

    @property
    def policy_shortfall_probability(self) -> float:
        return float(np.mean(self.policy_shortfall))


@dataclass(slots=True)
class _CertaintyEquivalentAccumulator:
    """Online weighted generalized mean for one scalar result per path."""

    paths: int
    elasticity: float
    total_weights: FloatArray = field(init=False)
    zero_with_positive_weight: BoolArray = field(init=False)
    weighted_log_sums: FloatArray | None = field(init=False)
    log_power_sums: FloatArray | None = field(init=False)
    power: float = field(init=False)

    def __post_init__(self) -> None:
        if not np.isfinite(self.elasticity) or self.elasticity <= 0:
            raise ValueError("elasticity must be positive")
        self.power = (
            0.0
            if np.isclose(self.elasticity, 1.0)
            else (self.elasticity - 1.0) / self.elasticity
        )
        self.total_weights = np.zeros(self.paths)
        self.zero_with_positive_weight = np.zeros(self.paths, dtype=bool)
        self.weighted_log_sums = np.zeros(self.paths) if self.power == 0.0 else None
        self.log_power_sums = (
            np.full(self.paths, -np.inf) if self.power != 0.0 else None
        )

    def ingest(self, values: FloatArray, weights: FloatArray) -> None:
        finite_values = np.asarray(values, dtype=float)
        finite_weights = np.asarray(weights, dtype=float)
        if finite_values.shape != (self.paths,) or finite_weights.shape != (self.paths,):
            raise ValueError("consumption and weights must be aligned arrays")
        if not np.all(np.isfinite(finite_values)) or not np.all(
            np.isfinite(finite_weights)
        ):
            raise ValueError("consumption and weights must contain only finite values")
        if np.any(finite_values < 0.0) or np.any(finite_weights < 0.0):
            raise ValueError("consumption and weights must be nonnegative")

        positive_weight = finite_weights > 0.0
        positive_value = finite_values > 0.0
        self.total_weights += finite_weights
        self.zero_with_positive_weight |= positive_weight & ~positive_value
        contributes = positive_weight & positive_value
        if not np.any(contributes):
            return
        if self.power == 0.0:
            assert self.weighted_log_sums is not None
            self.weighted_log_sums[contributes] += (
                finite_weights[contributes] * np.log(finite_values[contributes])
            )
            return
        assert self.log_power_sums is not None
        log_terms = (
            np.log(finite_weights[contributes])
            + self.power * np.log(finite_values[contributes])
        )
        self.log_power_sums[contributes] = np.logaddexp(
            self.log_power_sums[contributes],
            log_terms,
        )

    def finalize(self) -> FloatArray:
        result = np.zeros(self.paths)
        if self.power == 0.0:
            assert self.weighted_log_sums is not None
            eligible = (self.total_weights > 0.0) & ~self.zero_with_positive_weight
            result[eligible] = np.exp(
                self.weighted_log_sums[eligible] / self.total_weights[eligible]
            )
            return result

        assert self.log_power_sums is not None
        eligible = (self.total_weights > 0.0) & np.isfinite(self.log_power_sums)
        if self.power < 0.0:
            eligible &= ~self.zero_with_positive_weight
        result[eligible] = np.exp(
            (
                self.log_power_sums[eligible]
                - np.log(self.total_weights[eligible])
            )
            / self.power
        )
        return result


_TIME_VARYING_OUTCOMES = frozenset(
    {
        OutcomeType.SPENDING,
        OutcomeType.INSURED_BEQUEST,
        OutcomeType.LEVERAGE,
        OutcomeType.ALLOCATION_EQUITY,
        OutcomeType.RETIRED,
        OutcomeType.WORKING,
    }
)
_SIMULATION_OUTCOMES = frozenset(
    {
        *_TIME_VARYING_OUTCOMES,
        OutcomeType.TERMINAL_WEALTH,
        OutcomeType.BEQUEST,
        OutcomeType.RETIREMENT_AGE,
        OutcomeType.SOCIAL_SECURITY_CLAIM_AGE,
        OutcomeType.ANNUITIZATION_FRACTION,
        OutcomeType.RISK_TOLERANCE,
        OutcomeType.TIME_PREFERENCE,
    }
)
_STREAMABLE_ANNUAL_CURVE_TYPES = (
    LinearCurve,
    IsoelasticCurve,
    SpendingFloorCurve,
    TargetCurve,
)


def _can_stream_utility(utility_model: UtilityModel) -> bool:
    return all(
        addon.outcome not in _TIME_VARYING_OUTCOMES
        or type(addon.curve) in _STREAMABLE_ANNUAL_CURVE_TYPES
        for addon in utility_model.addons
    )


class _UtilityAccumulator:
    """Online utility aggregation and preference diagnostics."""

    __slots__ = (
        "_age_weights",
        "_aggregates",
        "_annual_addons",
        "_breach_counts",
        "_breach_methods",
        "_denominators",
        "_diagnostic_age_masks",
        "ages",
        "paths",
        "utility_model",
    )

    def __init__(
        self,
        utility_model: UtilityModel,
        ages: tuple[int, ...],
        paths: int,
    ) -> None:
        self.utility_model = utility_model
        self.ages = ages
        self.paths = paths
        utility_model.validate_outcomes(_SIMULATION_OUTCOMES)
        age_values = np.asarray(ages)
        offsets = age_values - utility_model.person.current_age
        base_weights = (
            (1.0 + utility_model.preferences.time_preference) ** -offsets
            * vitality(age_values, utility_model.preferences)
        )
        self._annual_addons = {
            addon.name: addon
            for addon in utility_model.addons
            if addon.outcome in _TIME_VARYING_OUTCOMES
        }
        self._age_weights: dict[str, FloatArray] = {}
        self._aggregates: dict[str, FloatArray] = {}
        self._denominators: dict[str, FloatArray] = {}
        self._breach_counts: dict[str, IntArray] = {}
        self._breach_methods: dict[str, object] = {}
        self._diagnostic_age_masks: dict[str, BoolArray] = {}
        for name, addon in self._annual_addons.items():
            age_mask = np.ones(len(ages), dtype=bool)
            if addon.minimum_age is not None:
                age_mask &= age_values >= addon.minimum_age
            if addon.maximum_age is not None:
                age_mask &= age_values <= addon.maximum_age
            self._age_weights[name] = np.asarray(
                base_weights * age_mask * addon.age_profile(age_values),
                dtype=np.float64,
            )
            initial = (
                np.full(paths, np.inf)
                if addon.aggregation is UtilityAggregation.WORST
                else np.zeros(paths)
            )
            self._aggregates[name] = initial
            if addon.aggregation is UtilityAggregation.DISCOUNTED_MEAN:
                self._denominators[name] = np.zeros(paths)
            breach_method = getattr(addon.curve, "diagnostic_breach", None)
            if callable(breach_method):
                self._breach_counts[name] = np.zeros(paths, dtype=np.int64)
                self._breach_methods[name] = breach_method
                self._diagnostic_age_masks[name] = age_mask

    def ingest(
        self,
        year: int,
        exposure: BoolArray,
        values: Mapping[OutcomeType, FloatArray],
    ) -> None:
        active_exposure = np.asarray(exposure, dtype=bool)
        if active_exposure.shape != (self.paths,):
            raise ValueError("utility exposure must align with simulated paths")
        for name, addon in self._annual_addons.items():
            raw_values = np.asarray(values[addon.outcome], dtype=float)
            if raw_values.shape != (self.paths,):
                raise ValueError(
                    f"decision outcome {addon.outcome!r} must align with simulated paths"
                )
            evaluated = np.asarray(addon.curve.evaluate(raw_values), dtype=float)
            if evaluated.shape != (self.paths,):
                raise ValueError(f"curve {addon.name!r} returned an invalid shape")
            if not np.all(np.isfinite(evaluated)):
                raise ValueError(
                    f"utility add-on {addon.name!r} score is not representable"
                )

            age_weight = self._age_weights[name][year]
            active = active_exposure & (age_weight > 0.0)
            aggregate = self._aggregates[name]
            with np.errstate(over="ignore", invalid="ignore"):
                if addon.aggregation is UtilityAggregation.DISCOUNTED_SUM:
                    aggregate[active] += age_weight * evaluated[active]
                elif addon.aggregation is UtilityAggregation.WORST:
                    aggregate[active] = np.minimum(
                        aggregate[active],
                        evaluated[active],
                    )
                elif addon.aggregation is UtilityAggregation.LAST:
                    aggregate[active] = evaluated[active]
                else:
                    aggregate[active] += age_weight * evaluated[active]
                    self._denominators[name][active] += age_weight

            if name in self._breach_counts:
                diagnostic_active = (
                    active_exposure & self._diagnostic_age_masks[name][year]
                )
                if not np.any(diagnostic_active):
                    continue
                breach_method = self._breach_methods[name]
                assert callable(breach_method)
                breached = np.asarray(breach_method(raw_values), dtype=bool)
                if breached.shape != (self.paths,):
                    raise ValueError(
                        f"curve {addon.name!r} diagnostic returned an invalid shape"
                    )
                self._breach_counts[name] += breached & diagnostic_active

    def finalize(
        self,
        scalar_values: Mapping[OutcomeType, float | FloatArray],
    ) -> tuple[dict[str, FloatArray], dict[str, PreferenceDiagnostic]]:
        component_scores: dict[str, FloatArray] = {}
        diagnostics: dict[str, PreferenceDiagnostic] = {}
        for addon in self.utility_model.addons:
            if addon.outcome in _TIME_VARYING_OUTCOMES:
                aggregated = self._aggregates[addon.name].copy()
                if addon.aggregation is UtilityAggregation.DISCOUNTED_MEAN:
                    denominator = self._denominators[addon.name]
                    aggregated = np.divide(
                        aggregated,
                        denominator,
                        out=np.zeros(self.paths),
                        where=denominator > 0.0,
                    )
                elif addon.aggregation is UtilityAggregation.WORST:
                    aggregated = np.where(np.isfinite(aggregated), aggregated, 0.0)
            else:
                raw_values = np.asarray(scalar_values[addon.outcome], dtype=float)
                if raw_values.ndim == 0:
                    raw_values = np.full(self.paths, float(raw_values))
                if raw_values.shape != (self.paths,):
                    raise ValueError(
                        f"decision outcome {addon.outcome!r} must align with simulated paths"
                    )
                aggregated = np.asarray(addon.curve.evaluate(raw_values), dtype=float)
                if aggregated.shape != (self.paths,):
                    raise ValueError(f"curve {addon.name!r} returned an invalid shape")
                breach_method = getattr(addon.curve, "diagnostic_breach", None)
                if callable(breach_method):
                    breached = np.asarray(breach_method(raw_values), dtype=bool)
                    if breached.ndim == 0:
                        breached = np.full(self.paths, bool(breached))
                    if breached.shape != (self.paths,):
                        raise ValueError(
                            f"curve {addon.name!r} diagnostic returned an invalid shape"
                        )
                    self._breach_counts[addon.name] = breached.astype(np.int64)

            with np.errstate(over="ignore", invalid="ignore"):
                score = np.asarray(addon.importance * aggregated, dtype=np.float64)
            if not np.all(np.isfinite(score)):
                raise ValueError(
                    f"utility add-on {addon.name!r} score is not representable"
                )
            component_scores[addon.name] = score
            if addon.name in self._breach_counts:
                diagnostics[addon.name] = PreferenceDiagnostic(
                    breach_count=self._breach_counts[addon.name],
                    utility_loss=np.maximum(-score, 0.0),
                )
        return component_scores, diagnostics


def _summary_payload(
    *,
    paths: int,
    seed: int,
    terminal_wealth: FloatArray,
    certainty_equivalents: FloatArray,
    utility_scores: FloatArray,
    margin_calls: IntArray,
    diagnostics: SimulationDiagnostics,
) -> dict[str, float | int]:
    summary: dict[str, float | int] = {
        "paths": paths,
        "seed": seed,
        "insolvency_probability": diagnostics.insolvency_probability,
        "policy_shortfall_probability": diagnostics.policy_shortfall_probability,
        "median_terminal_wealth": float(np.median(terminal_wealth)),
        "terminal_wealth_p05": float(np.quantile(terminal_wealth, 0.05)),
        "terminal_wealth_p95": float(np.quantile(terminal_wealth, 0.95)),
        "median_certainty_equivalent": float(np.median(certainty_equivalents)),
        "certainty_equivalent_p05": float(np.quantile(certainty_equivalents, 0.05)),
        "certainty_equivalent_p95": float(np.quantile(certainty_equivalents, 0.95)),
        "mean_utility": float(np.mean(utility_scores)),
        "median_utility": float(np.median(utility_scores)),
        "mean_margin_calls": float(np.mean(margin_calls)),
    }
    for name, diagnostic in diagnostics.preferences.items():
        summary[f"{name}_breach_probability"] = diagnostic.breach_probability
        summary[f"{name}_mean_breach_count"] = diagnostic.mean_breach_count
    return summary


@dataclass(frozen=True, slots=True)
class SimulationSummary:
    """Bounded-memory result retaining path-level scalar outcomes only."""

    paths: int
    seed: int
    terminal_wealth: FloatArray
    certainty_equivalents: FloatArray
    utility_scores: FloatArray
    utility_component_scores: Mapping[str, FloatArray]
    diagnostics: SimulationDiagnostics
    margin_calls: IntArray
    lifecycle_plan: LifecyclePlan

    def summary(self) -> dict[str, float | int]:
        return _summary_payload(
            paths=self.paths,
            seed=self.seed,
            terminal_wealth=self.terminal_wealth,
            certainty_equivalents=self.certainty_equivalents,
            utility_scores=self.utility_scores,
            margin_calls=self.margin_calls,
            diagnostics=self.diagnostics,
        )


@dataclass(frozen=True, slots=True)
class SimulationResult:
    ages: tuple[int, ...]
    wealth_paths: FloatArray
    spending_paths: FloatArray
    income_paths: FloatArray
    death_ages: IntArray
    diagnostics: SimulationDiagnostics
    margin_calls: IntArray
    certainty_equivalents: FloatArray
    utility_scores: FloatArray
    utility_component_scores: Mapping[str, FloatArray]
    decision_paths: Mapping[OutcomeType, FloatArray]
    lifecycle_plan: LifecyclePlan
    seed: int

    @property
    def insolvency_probability(self) -> float:
        return self.diagnostics.insolvency_probability

    @property
    def policy_shortfall_probability(self) -> float:
        return self.diagnostics.policy_shortfall_probability

    def summary(self) -> dict[str, float | int]:
        return _summary_payload(
            paths=len(self.wealth_paths),
            seed=self.seed,
            terminal_wealth=self.wealth_paths[:, -1],
            certainty_equivalents=self.certainty_equivalents,
            utility_scores=self.utility_scores,
            margin_calls=self.margin_calls,
            diagnostics=self.diagnostics,
        )


@dataclass(frozen=True, slots=True)
class _PreparedSimulation:
    scenario: PlanningScenario
    plan: LifecyclePlan
    utility_model: UtilityModel
    decision_policy: RollingDecisionPolicy
    mortality: GompertzMortality
    deterministic_income: FloatArray
    insurance_prices: FloatArray


@dataclass(slots=True)
class _ChunkAccumulator:
    paths: int
    seed: int
    lifecycle_plan: LifecyclePlan
    retain_paths: bool
    terminal_wealth: FloatArray = field(init=False)
    certainty_equivalents: FloatArray = field(init=False)
    utility_scores: FloatArray = field(init=False)
    margin_calls: IntArray = field(init=False)
    insolvent: BoolArray = field(init=False)
    policy_shortfall: BoolArray = field(init=False)
    utility_components: dict[str, FloatArray] = field(init=False, default_factory=dict)
    decision_paths: dict[OutcomeType, FloatArray] = field(init=False, default_factory=dict)
    preference_counts: dict[str, IntArray] = field(init=False, default_factory=dict)
    preference_losses: dict[str, FloatArray] = field(init=False, default_factory=dict)
    wealth_paths: FloatArray | None = field(init=False)
    spending_paths: FloatArray | None = field(init=False)
    income_paths: FloatArray | None = field(init=False)
    death_ages: IntArray | None = field(init=False)

    def __post_init__(self) -> None:
        horizon = len(self.lifecycle_plan.ages)
        self.terminal_wealth = np.empty(self.paths)
        self.certainty_equivalents = np.empty(self.paths)
        self.utility_scores = np.empty(self.paths)
        self.margin_calls = np.empty(self.paths, dtype=np.int64)
        self.insolvent = np.empty(self.paths, dtype=bool)
        self.policy_shortfall = np.empty(self.paths, dtype=bool)
        if self.retain_paths:
            self.wealth_paths = np.empty((self.paths, horizon + 1))
            self.spending_paths = np.empty((self.paths, horizon))
            self.income_paths = np.empty((self.paths, horizon))
            self.death_ages = np.empty(self.paths, dtype=np.int64)
        else:
            self.wealth_paths = None
            self.spending_paths = None
            self.income_paths = None
            self.death_ages = None

    def ingest(
        self,
        target: slice,
        result: SimulationResult | SimulationSummary,
    ) -> None:
        terminal_wealth = (
            result.wealth_paths[:, -1]
            if isinstance(result, SimulationResult)
            else result.terminal_wealth
        )
        self.terminal_wealth[target] = terminal_wealth
        self.certainty_equivalents[target] = result.certainty_equivalents
        self.utility_scores[target] = result.utility_scores
        self.margin_calls[target] = result.margin_calls
        self.insolvent[target] = result.diagnostics.insolvent
        self.policy_shortfall[target] = result.diagnostics.policy_shortfall
        for name, values in result.utility_component_scores.items():
            self.utility_components.setdefault(name, np.empty(self.paths))[target] = values
        for name, diagnostic in result.diagnostics.preferences.items():
            self.preference_counts.setdefault(
                name,
                np.empty(self.paths, dtype=np.int64),
            )[target] = diagnostic.breach_count
            self.preference_losses.setdefault(name, np.empty(self.paths))[target] = (
                diagnostic.utility_loss
            )
        if not self.retain_paths:
            return
        if not isinstance(result, SimulationResult):
            raise ValueError("full-path accumulation requires full simulation results")
        for name, values in result.decision_paths.items():
            self.decision_paths.setdefault(
                name,
                np.empty((self.paths, len(self.lifecycle_plan.ages))),
            )[target] = values
        assert self.wealth_paths is not None
        assert self.spending_paths is not None
        assert self.income_paths is not None
        assert self.death_ages is not None
        self.wealth_paths[target] = result.wealth_paths
        self.spending_paths[target] = result.spending_paths
        self.income_paths[target] = result.income_paths
        self.death_ages[target] = result.death_ages

    def build(self) -> SimulationResult | SimulationSummary:
        diagnostics = SimulationDiagnostics(
            insolvent=self.insolvent,
            policy_shortfall=self.policy_shortfall,
            preferences={
                name: PreferenceDiagnostic(
                    breach_count=counts,
                    utility_loss=self.preference_losses[name],
                )
                for name, counts in self.preference_counts.items()
            },
        )
        if not self.retain_paths:
            return SimulationSummary(
                paths=self.paths,
                seed=self.seed,
                terminal_wealth=self.terminal_wealth,
                certainty_equivalents=self.certainty_equivalents,
                utility_scores=self.utility_scores,
                utility_component_scores=self.utility_components,
                diagnostics=diagnostics,
                margin_calls=self.margin_calls,
                lifecycle_plan=self.lifecycle_plan,
            )
        assert self.wealth_paths is not None
        assert self.spending_paths is not None
        assert self.income_paths is not None
        assert self.death_ages is not None
        return SimulationResult(
            ages=self.lifecycle_plan.ages,
            wealth_paths=self.wealth_paths,
            spending_paths=self.spending_paths,
            income_paths=self.income_paths,
            death_ages=self.death_ages,
            diagnostics=diagnostics,
            margin_calls=self.margin_calls,
            certainty_equivalents=self.certainty_equivalents,
            utility_scores=self.utility_scores,
            utility_component_scores=self.utility_components,
            decision_paths=self.decision_paths,
            lifecycle_plan=self.lifecycle_plan,
            seed=self.seed,
        )


class MonteCarloEngine:
    def __init__(
        self,
        planner: LifecyclePlanner | None = None,
        market: MarketPathModel | None = None,
        tax_policy: TaxPolicy | None = None,
        leverage_terms: LeverageTerms | None = None,
        utility_addons: Sequence[UtilityAddon] = (),
        rolling_decision_policy: RollingDecisionPolicy | None = None,
        income_risk_model: IncomeRiskModel | None = None,
    ) -> None:
        self.planner = (
            LifecyclePlanner(utility_addons=utility_addons) if planner is None else planner
        )
        self.market = RegimeSwitchingMarket() if market is None else market
        self.tax_policy = TaxPolicy.for_2026() if tax_policy is None else tax_policy
        self.leverage_terms = LeverageTerms() if leverage_terms is None else leverage_terms
        self.utility_addons = (
            tuple(utility_addons) if utility_addons else tuple(self.planner.utility_addons)
        )
        self.rolling_decision_policy = rolling_decision_policy
        self.income_risk_model = income_risk_model

    def _after_tax_income(
        self,
        scenario: PlanningScenario,
        plan: LifecyclePlan,
    ) -> FloatArray:
        salary = self.planner.salary_model.project(scenario.person, scenario.income)
        result = np.zeros(scenario.person.horizon)
        for year, age in enumerate(plan.ages):
            wages = salary[year]
            social_security = max(plan.income_path[year] - wages, 0.0)
            taxes = self.tax_policy.calculate(
                wages=wages,
                social_security=social_security,
                include_payroll=wages > 0,
            )
            employer_match = (
                scenario.income.employer_match
                if age < scenario.person.retirement_age
                else 0.0
            )
            result[year] = wages + social_security - taxes.total + employer_match
        return result

    @staticmethod
    def _validate_spending_target(
        target: FloatArray,
        expected_shape: tuple[int, ...],
    ) -> FloatArray:
        try:
            validated = np.asarray(target, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("spending target must be numeric") from error
        if validated.shape != expected_shape:
            raise ValueError("spending target must align with simulated paths")
        if not np.all(np.isfinite(validated)):
            raise ValueError("spending target must be finite")
        if np.any(validated < 0):
            raise ValueError("spending target cannot be negative")
        return validated

    def _portfolio_return(
        self,
        market: MarketPaths,
        year: int,
        allocation: Allocation | FloatArray,
        effective_leverage: FloatArray,
        instrument: LeverageInstrument,
    ) -> FloatArray:
        return leveraged_portfolio_return(
            market,
            year,
            allocation,
            effective_leverage,
            instrument,
            market_config=self.market.config,
            leverage_terms=self.leverage_terms,
        )

    def _expected_leverage_cost(
        self,
        instrument: LeverageInstrument,
    ) -> float:
        terms = self.leverage_terms
        if instrument is LeverageInstrument.FUTURES:
            direct_cost = (
                terms.futures_financing_spread
                + terms.futures_roll_cost
            )
            expected_positive_overlay = max(
                self.market.config.equity_risk_premium - direct_cost,
                0.0,
            )
            return (
                direct_cost
                + terms.futures_tax_rate * expected_positive_overlay
            )
        if instrument is LeverageInstrument.BOX_SPREAD:
            return (
                terms.box_financing_spread
                + terms.box_dividend_yield * terms.box_dividend_tax_rate
            )
        return self.market.config.margin_spread

    @staticmethod
    def _certainty_equivalents(
        spending: FloatArray,
        alive: BoolArray,
        scenario: PlanningScenario,
    ) -> FloatArray:
        preferences = scenario.preferences
        ages = np.arange(
            scenario.person.current_age,
            scenario.person.maximum_age + 1,
        )
        base_weights = (1.0 + preferences.time_preference) ** -np.arange(len(ages)) * vitality(
            ages, preferences
        )
        weights = alive * base_weights[None, :]
        return _weighted_certainty_equivalent(
            spending,
            weights,
            preferences.consumption_elasticity,
        )

    @staticmethod
    def _preference_diagnostics(
        utility_model: UtilityModel,
        outcome: UtilityOutcome,
        component_scores: Mapping[str, FloatArray],
    ) -> dict[str, PreferenceDiagnostic]:
        diagnostics: dict[str, PreferenceDiagnostic] = {}
        ages = np.asarray(outcome.ages)
        for addon in utility_model.addons:
            if addon.outcome not in outcome.available_outcomes:
                continue
            breach_method = getattr(addon.curve, "diagnostic_breach", None)
            if not callable(breach_method):
                continue
            breached = np.asarray(
                breach_method(outcome.values(addon.outcome)),
                dtype=bool,
            )
            if breached.ndim == 2:
                active = outcome.exposure > 0
                if addon.minimum_age is not None:
                    active &= ages[np.newaxis, :] >= addon.minimum_age
                if addon.maximum_age is not None:
                    active &= ages[np.newaxis, :] <= addon.maximum_age
                breach_count = np.sum(breached & active, axis=1, dtype=np.int64)
            else:
                breach_count = breached.astype(np.int64)
            diagnostics[addon.name] = PreferenceDiagnostic(
                breach_count=np.asarray(breach_count, dtype=np.int64),
                utility_loss=np.maximum(-component_scores[addon.name], 0.0),
            )
        return diagnostics

    def _prepare(self, scenario: PlanningScenario) -> _PreparedSimulation:
        plan = self.planner.plan(scenario)
        utility_model = UtilityModel.from_scenario(scenario, self.utility_addons)
        mortality = GompertzMortality.from_person(scenario.person)
        decision_policy = (
            JointRollingDecisionOptimizer(utility_model)
            if self.rolling_decision_policy is None
            else self.rolling_decision_policy
        )
        return _PreparedSimulation(
            scenario=scenario,
            plan=plan,
            utility_model=utility_model,
            decision_policy=decision_policy,
            mortality=mortality,
            deterministic_income=self._after_tax_income(scenario, plan),
            insurance_prices=mortality.permanent_insurance_price_curve(
                self.planner.capital_markets.risk_free_rate,
            ),
        )

    def simulate(
        self,
        scenario: PlanningScenario | None = None,
        *,
        settings: SimulationSettings | None = None,
        spending_policy: SpendingPolicy | None = None,
        allocation_override: Allocation | None = None,
    ) -> SimulationResult:
        scenario = PlanningScenario() if scenario is None else scenario
        settings = SimulationSettings() if settings is None else settings
        prepared = self._prepare(scenario)
        result = self._simulate_prepared(
            prepared,
            settings,
            spending_policy=spending_policy,
            allocation_override=allocation_override,
        )
        assert isinstance(result, SimulationResult)
        return result

    def _simulate_prepared(
        self,
        prepared: _PreparedSimulation,
        settings: SimulationSettings,
        *,
        seed: int | np.random.SeedSequence | None = None,
        spending_policy: SpendingPolicy | None = None,
        allocation_override: Allocation | None = None,
        retain_paths: bool = True,
    ) -> SimulationResult | SimulationSummary:
        if not retain_paths and not _can_stream_utility(prepared.utility_model):
            full_result = self._simulate_prepared(
                prepared,
                settings,
                seed=seed,
                spending_policy=spending_policy,
                allocation_override=allocation_override,
                retain_paths=True,
            )
            assert isinstance(full_result, SimulationResult)
            return SimulationSummary(
                paths=len(full_result.wealth_paths),
                seed=full_result.seed,
                terminal_wealth=full_result.wealth_paths[:, -1].copy(),
                certainty_equivalents=full_result.certainty_equivalents,
                utility_scores=full_result.utility_scores,
                utility_component_scores=full_result.utility_component_scores,
                diagnostics=full_result.diagnostics,
                margin_calls=full_result.margin_calls,
                lifecycle_plan=full_result.lifecycle_plan,
            )
        scenario = prepared.scenario
        plan = prepared.plan
        utility_model = prepared.utility_model
        decision_policy = prepared.decision_policy

        paths = settings.paths
        horizon = scenario.person.horizon
        simulation_seed = settings.seed if seed is None else seed
        market = self.market.generate(
            paths=paths,
            horizon=horizon,
            seed=simulation_seed,
            antithetic=settings.antithetic,
        )
        seed_sequence = (
            simulation_seed
            if isinstance(simulation_seed, np.random.SeedSequence)
            else np.random.SeedSequence(simulation_seed)
        )
        rng = np.random.default_rng(seed_sequence.spawn(1)[0])
        mortality = prepared.mortality
        death_ages = (
            mortality.sample_death_ages(paths, rng)
            if settings.stochastic_lifespan
            else np.full(paths, scenario.person.maximum_age + 1, dtype=np.int64)
        )

        deterministic_income = prepared.deterministic_income
        income_risk_model: IncomeRiskModel | None = None
        income_risk_state: IncomeRiskState | None = None
        if settings.stochastic_income:
            income_risk_model = self.income_risk_model
            if income_risk_model is None:
                income_risk_model = TransitoryMarketJobLoss(
                    baseline_probability=settings.job_loss_probability,
                    market_sensitivity=settings.job_loss_market_sensitivity,
                    income_fraction=settings.job_loss_income_fraction,
                    probability_cap=0.50,
                )
            income_risk_state = income_risk_model.initial_state(paths)
            _validate_income_risk_state_paths(income_risk_state, paths)
        wealth = np.zeros((paths, horizon + 1)) if retain_paths else None
        spending = np.zeros((paths, horizon)) if retain_paths else None
        income = np.zeros((paths, horizon)) if retain_paths else None
        alive_history = (
            np.zeros((paths, horizon), dtype=bool) if retain_paths else None
        )
        equity_history = np.zeros((paths, horizon)) if retain_paths else None
        leverage_history = np.ones((paths, horizon)) if retain_paths else None
        insured_bequest_history = (
            np.zeros((paths, horizon)) if retain_paths else None
        )
        if wealth is None:
            current_wealth = np.full(paths, scenario.wealth.total)
        else:
            wealth[:, 0] = scenario.wealth.total
            current_wealth = wealth[:, 0]
        certainty_accumulator = (
            None
            if retain_paths
            else _CertaintyEquivalentAccumulator(
                paths,
                scenario.preferences.consumption_elasticity,
            )
        )
        utility_accumulator = (
            None
            if retain_paths
            else _UtilityAccumulator(utility_model, plan.ages, paths)
        )
        if retain_paths:
            certainty_age_weights = None
        else:
            ages = np.asarray(plan.ages)
            certainty_age_weights = (
                (1.0 + scenario.preferences.time_preference) ** -np.arange(horizon)
                * vitality(ages, scenario.preferences)
            )
        insured_bequest = np.zeros(paths)
        policy_shortfall = np.zeros(paths, dtype=bool)
        insolvent = np.zeros(paths, dtype=bool)
        margin_calls = np.zeros(paths, dtype=np.int64)
        effective_leverage = np.ones(paths)
        cooldown = np.zeros(paths, dtype=np.int64)

        for year, age in enumerate(plan.ages):
            if wealth is not None:
                current_wealth = wealth[:, year]
            alive = age < death_ages
            newly_dead = age == death_ages
            current_wealth += np.where(newly_dead, insured_bequest, 0.0)
            insured_bequest = np.where(newly_dead, 0.0, insured_bequest)
            if alive_history is not None:
                alive_history[:, year] = alive
            realized_income = np.full(paths, deterministic_income[year])
            if income_risk_model is not None and age < scenario.person.retirement_age:
                current_excess_return = (
                    market.equity_returns[:, year]
                    - market.real_rates[:, year]
                    - self.market.config.equity_risk_premium
                )
                lagged_excess_return = (
                    np.zeros(paths, dtype=float)
                    if year == 0
                    else market.equity_returns[:, year - 1]
                    - market.real_rates[:, year - 1]
                    - self.market.config.equity_risk_premium
                )
                assert income_risk_state is not None
                income_risk_step = income_risk_model.transition(
                    IncomeRiskContext(
                        year=year,
                        deterministic_income=realized_income,
                        working=np.full(paths, True, dtype=bool),
                        current_excess_return=current_excess_return,
                        lagged_excess_return=lagged_excess_return,
                        random_uniform=rng.random(paths),
                    ),
                    income_risk_state,
                )
                _validate_income_risk_step_paths(income_risk_step, paths)
                income_risk_state = income_risk_step.state
                realized_income = income_risk_step.realized_income
            realized_income = np.where(alive, realized_income, 0.0)
            if income is not None:
                income[:, year] = realized_income

            context = SpendingContext(
                year=year,
                wealth=current_wealth,
                income=realized_income,
                future_income=deterministic_income,
                real_rate=market.real_rates[:, year],
                scenario=scenario,
                lifecycle_plan=plan,
            )
            available = np.maximum(current_wealth + realized_income, 0.0)
            custom_spending = None
            if spending_policy is not None:
                custom_spending = self._validate_spending_target(
                    spending_policy.target(context),
                    context.wealth.shape,
                )
            base_allocation = (
                plan.glide_path[year] if allocation_override is None else allocation_override
            )
            insurance_price = prepared.insurance_prices[year]
            decision = decision_policy.decide(
                RollingDecisionContext(
                    spending=context,
                    existing_insured_bequest=insured_bequest,
                    insurance_price=insurance_price,
                    maximum_leverage=settings.leverage,
                    market=self.market.config,
                    base_allocation=base_allocation,
                    fixed_allocation=allocation_override is not None,
                    leverage_cost=self._expected_leverage_cost(
                        settings.leverage_instrument,
                    ),
                    effective_leverage=effective_leverage,
                    leverage_locked=cooldown > 0,
                    active_paths=alive,
                    fixed_spending=custom_spending,
                )
            )
            premium_limit = (
                available
                if custom_spending is None
                else np.maximum(available - custom_spending, 0.0)
            )
            premium = np.where(
                alive,
                np.minimum(decision.insurance_premium, premium_limit),
                0.0,
            )
            insured_bequest = np.where(
                alive & (insurance_price > 0),
                insured_bequest + premium / max(insurance_price, np.finfo(float).tiny),
                insured_bequest,
            )
            available_after_premium = np.maximum(available - premium, 0.0)
            desired_spending = (
                decision.spending
                if custom_spending is None
                else custom_spending
            )
            target = np.where(alive, desired_spending, 0.0)
            actual = np.minimum(target, available_after_premium)
            if spending is not None:
                spending[:, year] = actual
            policy_shortfall |= alive & (actual + 1e-8 < target)
            insolvent |= alive & (current_wealth <= 1e-8) & (realized_income <= 1e-8)
            investable = np.maximum(available_after_premium - actual, 0.0)
            actual_equity = (
                base_allocation.equity
                if allocation_override is not None
                else decision.equity_fraction
            )
            annual_equity = np.where(
                alive,
                actual_equity,
                0.0,
            )
            if equity_history is not None:
                equity_history[:, year] = annual_equity
            if insured_bequest_history is not None:
                insured_bequest_history[:, year] = insured_bequest
            desired_leverage = np.where(alive, decision.leverage, 1.0)
            effective_leverage = np.where(
                cooldown == 0,
                desired_leverage,
                effective_leverage,
            )
            if leverage_history is not None:
                leverage_history[:, year] = effective_leverage
            if certainty_accumulator is not None:
                assert certainty_age_weights is not None
                certainty_accumulator.ingest(
                    actual,
                    alive * certainty_age_weights[year],
                )
            if utility_accumulator is not None:
                retired = np.full(
                    paths,
                    float(age >= scenario.person.retirement_age),
                )
                utility_accumulator.ingest(
                    year,
                    alive,
                    {
                        OutcomeType.SPENDING: actual,
                        OutcomeType.INSURED_BEQUEST: insured_bequest,
                        OutcomeType.LEVERAGE: effective_leverage,
                        OutcomeType.ALLOCATION_EQUITY: annual_equity,
                        OutcomeType.RETIRED: retired,
                        OutcomeType.WORKING: 1.0 - retired,
                    },
                )
            allocation: Allocation | FloatArray = (
                base_allocation if allocation_override is not None else decision.equity_fraction
            )
            portfolio_return = self._portfolio_return(
                market,
                year,
                allocation,
                effective_leverage,
                settings.leverage_instrument,
            )
            next_wealth = investable * (1.0 + portfolio_return)

            levered = effective_leverage > 1.0
            debt = investable * np.maximum(effective_leverage - 1.0, 0.0)
            gross_assets = next_wealth + debt
            equity_ratio = np.divide(
                next_wealth,
                gross_assets,
                out=np.zeros_like(next_wealth),
                where=gross_assets > 0,
            )
            called = (
                alive
                & levered
                & (settings.maintenance_margin > 0)
                & (equity_ratio < settings.maintenance_margin)
            )
            margin_calls += called
            next_wealth = np.maximum(next_wealth, 0.0)

            cooldown = np.maximum(cooldown - 1, 0)
            effective_leverage = np.where(called, settings.margin_call_leverage, effective_leverage)
            cooldown = np.where(called, settings.margin_call_cooldown_years, cooldown)
            updated_wealth = np.where(alive, next_wealth, current_wealth)
            if wealth is None:
                current_wealth = updated_wealth
            else:
                wealth[:, year + 1] = updated_wealth

        if not retain_paths:
            assert certainty_accumulator is not None
            assert utility_accumulator is not None
            terminal_wealth = current_wealth + insured_bequest
            certainty_equivalents = certainty_accumulator.finalize()
            utility_components, preference_diagnostics = utility_accumulator.finalize(
                {
                    OutcomeType.TERMINAL_WEALTH: terminal_wealth,
                    OutcomeType.BEQUEST: terminal_wealth,
                    OutcomeType.RETIREMENT_AGE: float(scenario.person.retirement_age),
                    OutcomeType.SOCIAL_SECURITY_CLAIM_AGE: float(
                        scenario.person.claiming_age
                    ),
                    OutcomeType.ANNUITIZATION_FRACTION: (
                        scenario.preferences.annuitization_fraction
                    ),
                    OutcomeType.RISK_TOLERANCE: (
                        scenario.preferences.effective_risk_tolerance
                    ),
                    OutcomeType.TIME_PREFERENCE: scenario.preferences.time_preference,
                }
            )
            utility_scores = np.sum(
                np.stack(tuple(utility_components.values())),
                axis=0,
            )
            return SimulationSummary(
                paths=paths,
                seed=settings.seed,
                terminal_wealth=terminal_wealth,
                certainty_equivalents=certainty_equivalents,
                utility_scores=utility_scores,
                utility_component_scores=utility_components,
                diagnostics=SimulationDiagnostics(
                    insolvent=insolvent,
                    policy_shortfall=policy_shortfall,
                    preferences=preference_diagnostics,
                ),
                margin_calls=margin_calls,
                lifecycle_plan=plan,
            )

        assert wealth is not None
        assert spending is not None
        assert income is not None
        assert alive_history is not None
        assert equity_history is not None
        assert leverage_history is not None
        assert insured_bequest_history is not None
        wealth[:, -1] += insured_bequest
        certainty_equivalents = self._certainty_equivalents(
            spending,
            alive_history,
            scenario,
        )
        retired_by_year = np.asarray(plan.ages, dtype=float) >= scenario.person.retirement_age
        retired = np.broadcast_to(retired_by_year, spending.shape).astype(float)
        working = 1.0 - retired
        utility_outcome = UtilityOutcome(
            spending=spending,
            exposure=alive_history.astype(float),
            ages=plan.ages,
            terminal_wealth=wealth[:, -1],
            decisions={
                OutcomeType.BEQUEST: wealth[:, -1],
                OutcomeType.RETIREMENT_AGE: float(scenario.person.retirement_age),
                OutcomeType.SOCIAL_SECURITY_CLAIM_AGE: float(scenario.person.claiming_age),
                OutcomeType.ANNUITIZATION_FRACTION: (scenario.preferences.annuitization_fraction),
                OutcomeType.INSURED_BEQUEST: insured_bequest_history,
                OutcomeType.LEVERAGE: leverage_history,
                OutcomeType.ALLOCATION_EQUITY: equity_history,
                OutcomeType.RISK_TOLERANCE: (scenario.preferences.effective_risk_tolerance),
                OutcomeType.TIME_PREFERENCE: scenario.preferences.time_preference,
                OutcomeType.RETIRED: retired,
                OutcomeType.WORKING: working,
            },
        )
        utility_model.validate_outcomes(utility_outcome.available_outcomes)
        utility_components = utility_model.decompose(utility_outcome)
        utility_scores = np.sum(
            np.stack(tuple(utility_components.values())),
            axis=0,
        )
        diagnostics = SimulationDiagnostics(
            insolvent=insolvent,
            policy_shortfall=policy_shortfall,
            preferences=self._preference_diagnostics(
                utility_model,
                utility_outcome,
                utility_components,
            ),
        )
        return SimulationResult(
            ages=plan.ages,
            wealth_paths=wealth,
            spending_paths=spending,
            income_paths=income,
            death_ages=death_ages,
            diagnostics=diagnostics,
            margin_calls=margin_calls,
            certainty_equivalents=certainty_equivalents,
            utility_scores=utility_scores,
            utility_component_scores=utility_components,
            decision_paths={
                OutcomeType.LEVERAGE: leverage_history,
                OutcomeType.ALLOCATION_EQUITY: equity_history,
                OutcomeType.INSURED_BEQUEST: insured_bequest_history,
            },
            lifecycle_plan=plan,
            seed=settings.seed,
        )

    @overload
    def simulate_chunked(
        self,
        scenario: PlanningScenario | None = None,
        *,
        settings: SimulationSettings | None = None,
        chunk_size: int = 10_000,
        retain_paths: Literal[True],
        spending_policy: SpendingPolicy | None = None,
        allocation_override: Allocation | None = None,
    ) -> SimulationResult: ...

    @overload
    def simulate_chunked(
        self,
        scenario: PlanningScenario | None = None,
        *,
        settings: SimulationSettings | None = None,
        chunk_size: int = 10_000,
        retain_paths: Literal[False] = False,
        spending_policy: SpendingPolicy | None = None,
        allocation_override: Allocation | None = None,
    ) -> SimulationSummary: ...

    def simulate_chunked(
        self,
        scenario: PlanningScenario | None = None,
        *,
        settings: SimulationSettings | None = None,
        chunk_size: int = 10_000,
        retain_paths: bool = False,
        spending_policy: SpendingPolicy | None = None,
        allocation_override: Allocation | None = None,
    ) -> SimulationResult | SimulationSummary:
        """Run independent deterministic chunks with bounded temporary memory.

        Scalar path outcomes are retained for exact quantiles. Full path
        matrices are allocated only when ``retain_paths`` is true.
        """
        scenario = PlanningScenario() if scenario is None else scenario
        settings = SimulationSettings() if settings is None else settings
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        prepared = self._prepare(scenario)
        accumulator = _ChunkAccumulator(
            paths=settings.paths,
            seed=settings.seed,
            lifecycle_plan=prepared.plan,
            retain_paths=retain_paths,
        )
        if chunk_size >= settings.paths:
            accumulator.ingest(
                slice(0, settings.paths),
                self._simulate_prepared(
                    prepared,
                    settings,
                    spending_policy=spending_policy,
                    allocation_override=allocation_override,
                    retain_paths=retain_paths,
                ),
            )
            result = accumulator.build()
            if retain_paths:
                assert isinstance(result, SimulationResult)
            else:
                assert isinstance(result, SimulationSummary)
            return result

        paths = settings.paths
        chunk_count = (paths + chunk_size - 1) // chunk_size
        child_sequences = np.random.SeedSequence(settings.seed).spawn(chunk_count)
        offset = 0
        for sequence in child_sequences:
            count = min(chunk_size, paths - offset)
            accumulator.ingest(
                slice(offset, offset + count),
                self._simulate_prepared(
                    prepared,
                    replace(settings, paths=count),
                    seed=sequence,
                    spending_policy=spending_policy,
                    allocation_override=allocation_override,
                    retain_paths=retain_paths,
                ),
            )
            offset += count

        result = accumulator.build()
        if retain_paths:
            assert isinstance(result, SimulationResult)
        else:
            assert isinstance(result, SimulationSummary)
        return result
