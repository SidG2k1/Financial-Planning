"""Isoelastic consumption, vitality, certainty-equivalent, and bequest utility."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .domain import (
    OutcomeType,
    Person,
    PlanningScenario,
    Preferences,
    UtilityAddonConfig,
    UtilityAggregation,
    UtilityCurveKind,
)

FloatArray = NDArray[np.float64]


def _finite_number(value: float, name: str) -> None:
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _finite_elasticity(elasticity: float) -> None:
    _finite_number(elasticity, "elasticity")
    if elasticity <= 0:
        raise ValueError("elasticity must be positive")


def _finite_values(value: ArrayLike, name: str) -> FloatArray:
    values = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return np.asarray(values, dtype=np.float64)


def isoelastic_utility(value: ArrayLike, elasticity: float) -> FloatArray:
    """Workbook utility parameterized by elasticity of intertemporal substitution."""
    _finite_elasticity(elasticity)
    values = _finite_values(value, "utility values")
    if np.any(values <= 0):
        raise ValueError("utility is defined only for positive values")
    if np.isclose(elasticity, 1.0):
        return np.asarray(np.log(values), dtype=np.float64)
    power = (elasticity - 1.0) / elasticity
    return np.asarray((np.power(values, power) - 1.0) / power, dtype=np.float64)


def inverse_isoelastic_utility(utility: ArrayLike, elasticity: float) -> FloatArray:
    _finite_elasticity(elasticity)
    values = _finite_values(utility, "utility values")
    if np.isclose(elasticity, 1.0):
        return np.asarray(np.exp(values), dtype=np.float64)
    power = (elasticity - 1.0) / elasticity
    base = power * values + 1.0
    if np.any(base <= 0):
        raise ValueError("utility value is outside the inverse function domain")
    return np.asarray(np.power(base, 1.0 / power), dtype=np.float64)


def _weighted_certainty_equivalent(
    value: ArrayLike,
    weights: ArrayLike,
    elasticity: float,
) -> FloatArray:
    """Return an exact weighted generalized mean along the final axis."""
    _finite_elasticity(elasticity)
    values = _finite_values(value, "consumption")
    finite_weights = _finite_values(weights, "certainty-equivalent weights")
    if values.ndim == 0 or values.shape != finite_weights.shape:
        raise ValueError("consumption and weights must be aligned arrays")
    if np.any(values < 0.0) or np.any(finite_weights < 0.0):
        raise ValueError("consumption and weights must be nonnegative")

    flat_values = values.reshape(-1, values.shape[-1])
    flat_weights = finite_weights.reshape(-1, finite_weights.shape[-1])
    total_weights = np.sum(flat_weights, axis=1)
    result = np.zeros(len(flat_values))
    positive_weight = flat_weights > 0.0
    zero_with_weight = np.any(positive_weight & (flat_values == 0.0), axis=1)
    power = (
        0.0
        if np.isclose(elasticity, 1.0)
        else (elasticity - 1.0) / elasticity
    )

    if power == 0.0:
        eligible = (total_weights > 0.0) & ~zero_with_weight
        if np.any(eligible):
            selected_values = flat_values[eligible]
            selected_weights = flat_weights[eligible]
            log_values = np.full_like(selected_values, -np.inf)
            np.log(
                selected_values,
                out=log_values,
                where=selected_values > 0.0,
            )
            weighted_logs = np.zeros_like(selected_values)
            np.multiply(
                selected_weights,
                log_values,
                out=weighted_logs,
                where=selected_weights > 0.0,
            )
            result[eligible] = np.exp(
                np.sum(weighted_logs, axis=1) / total_weights[eligible]
            )
        return np.asarray(result.reshape(values.shape[:-1]), dtype=np.float64)

    has_positive_value = np.any(
        positive_weight & (flat_values > 0.0),
        axis=1,
    )
    eligible = (total_weights > 0.0) & has_positive_value
    if power < 0.0:
        eligible &= ~zero_with_weight
    if np.any(eligible):
        selected_values = flat_values[eligible]
        selected_weights = flat_weights[eligible]
        log_terms = np.full_like(selected_values, -np.inf)
        contributes = (selected_weights > 0.0) & (selected_values > 0.0)
        log_terms[contributes] = (
            np.log(selected_weights[contributes])
            + power * np.log(selected_values[contributes])
        )
        maximum_log = np.max(log_terms, axis=1)
        mean_power_log = (
            maximum_log
            + np.log(
                np.sum(
                    np.exp(log_terms - maximum_log[:, np.newaxis]),
                    axis=1,
                )
            )
            - np.log(total_weights[eligible])
        )
        result[eligible] = np.exp(mean_power_log / power)
    return np.asarray(result.reshape(values.shape[:-1]), dtype=np.float64)


def vitality(age: ArrayLike, preferences: Preferences) -> FloatArray:
    ages = np.asarray(age, dtype=float)
    x = (ages - preferences.vitality_peak_age) / preferences.vitality_half_life
    result = preferences.vitality_floor + (1.0 - preferences.vitality_floor) * np.exp(-(x**2))
    return np.asarray(
        np.where(ages <= preferences.vitality_peak_age, 1.0, result),
        dtype=np.float64,
    )


def consumption_growth_rate(certainty_equivalent_return: float, preferences: Preferences) -> float:
    """Workbook discretionary-consumption growth rate before survival rescheduling."""
    return float(
        ((1.0 + certainty_equivalent_return) / (1.0 + preferences.time_preference))
        ** preferences.consumption_elasticity
        - 1.0
    )


class UtilityCurve(Protocol):
    """A scalar preference curve expressed in dimensionless utility points."""

    def evaluate(self, value: ArrayLike) -> FloatArray:
        """Evaluate one or more outcome values."""


class DifferentiableUtilityCurve(UtilityCurve, Protocol):
    """A curve with an analytic marginal utility and finite breakpoints."""

    def marginal_utility(self, value: ArrayLike) -> FloatArray:
        """Return d utility / d outcome."""

    def breakpoints(self) -> tuple[float, ...]:
        """Return values at which the derivative may change regime."""


@dataclass(frozen=True, slots=True)
class LinearCurve:
    """A constant marginal value for status flows and other cardinal outcomes."""

    slope: float = 1.0
    intercept: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.slope) or not np.isfinite(self.intercept):
            raise ValueError("slope and intercept must be finite")

    def evaluate(self, value: ArrayLike) -> FloatArray:
        values = np.asarray(value, dtype=float)
        result = np.empty_like(values, dtype=np.float64)
        self.evaluate_into(values, result)
        return result

    def evaluate_into(self, value: ArrayLike, out: FloatArray) -> None:
        np.multiply(_finite_values(value, "value"), self.slope, out=out)
        out += self.intercept

    def marginal_utility(self, value: ArrayLike) -> FloatArray:
        values = np.asarray(value, dtype=float)
        result = np.empty_like(values, dtype=np.float64)
        self.marginal_utility_into(values, result)
        return result

    def marginal_utility_into(self, value: ArrayLike, out: FloatArray) -> None:
        _finite_values(value, "value")
        out.fill(self.slope)

    def breakpoints(self) -> tuple[float, ...]:
        return ()


@dataclass(frozen=True, slots=True)
class IsoelasticCurve:
    """Bounded CRRA utility anchored at zero when ``value == reference``.

    The finite lower bound keeps this preference soft: even a zero outcome can
    be traded against sufficiently important competing goals. Below the
    reference, an algebraic transform preserves the reference marginal while
    approaching the lower bound without a constant-utility interval.
    """

    reference: float
    elasticity: float
    minimum_utility: float = -10.0
    _power: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _finite_number(self.reference, "reference")
        _finite_number(self.elasticity, "elasticity")
        _finite_number(self.minimum_utility, "minimum_utility")
        if self.reference <= 0:
            raise ValueError("reference must be positive")
        if self.elasticity <= 0:
            raise ValueError("elasticity must be positive")
        if self.minimum_utility > 0:
            raise ValueError("minimum_utility cannot be positive")
        power = (
            0.0 if np.isclose(self.elasticity, 1.0) else (self.elasticity - 1.0) / self.elasticity
        )
        object.__setattr__(self, "_power", power)

    def evaluate(self, value: ArrayLike) -> FloatArray:
        values = np.asarray(value, dtype=float)
        result = np.empty_like(values, dtype=np.float64)
        self.evaluate_into(values, result)
        return result

    def evaluate_into(self, value: ArrayLike, out: FloatArray) -> None:
        values = _finite_values(value, "value")
        normalized = np.maximum(values / self.reference, np.finfo(float).eps)
        if self._power == 0:
            np.log(normalized, out=out)
        else:
            effective_power = np.full_like(normalized, self._power)
            effective_power[normalized < 1.0] = -abs(self._power)
            with np.errstate(over="ignore"):
                np.power(normalized, effective_power, out=out)
            out -= 1.0
            out /= effective_power
        below_reference = out < 0.0
        if self.minimum_utility == 0.0:
            out[below_reference] = 0.0
            return
        lower_scale = -self.minimum_utility
        distance = -out[below_reference] / lower_scale
        out[below_reference] = (
            self.minimum_utility
            + lower_scale / (1.0 + distance)
        )
        out[values <= 0.0] = self.minimum_utility

    def marginal_utility(self, value: ArrayLike) -> FloatArray:
        values = np.asarray(value, dtype=float)
        result = np.empty_like(values, dtype=np.float64)
        self.marginal_utility_into(values, result)
        return result

    def marginal_utility_into(self, value: ArrayLike, out: FloatArray) -> None:
        values = _finite_values(value, "value")
        positive = values > 0.0
        out.fill(0.0)
        if not np.any(positive):
            return
        normalized = values[positive] / self.reference
        log_normalized = np.log(normalized)
        if self._power == 0:
            raw = log_normalized
            effective_power = np.zeros_like(normalized)
        else:
            effective_power = np.full_like(normalized, self._power)
            effective_power[normalized < 1.0] = -abs(self._power)
            with np.errstate(over="ignore"):
                raw = (
                    np.power(normalized, effective_power) - 1.0
                ) / effective_power
        log_marginal = (effective_power - 1.0) * log_normalized - np.log(self.reference)
        below_reference = raw < 0.0
        if self.minimum_utility == 0.0:
            log_marginal[below_reference] = -np.inf
        else:
            distance = -raw[below_reference] / -self.minimum_utility
            log_marginal[below_reference] -= 2.0 * np.log1p(distance)
        out[positive] = np.exp(
            np.minimum(log_marginal, np.log(np.finfo(float).max))
        )

    def breakpoints(self) -> tuple[float, ...]:
        return (self.reference,)


@dataclass(frozen=True, slots=True)
class SpendingFloorCurve:
    """Zero above a desired floor and progressively worse below it.

    ``importance`` belongs on :class:`UtilityAddon`. ``scale`` controls how
    quickly the preference becomes important: at ``threshold - scale`` this
    curve contributes -1 before weighting.
    """

    threshold: float
    scale: float
    curvature: float = 2.0

    def __post_init__(self) -> None:
        _finite_number(self.threshold, "threshold")
        _finite_number(self.scale, "scale")
        _finite_number(self.curvature, "curvature")
        if self.threshold < 0:
            raise ValueError("threshold must be nonnegative")
        if self.scale <= 0:
            raise ValueError("scale must be positive")
        if self.curvature < 1:
            raise ValueError("curvature must be at least one")

    def evaluate(self, value: ArrayLike) -> FloatArray:
        values = np.asarray(value, dtype=float)
        result = np.empty_like(values, dtype=np.float64)
        self.evaluate_into(values, result)
        return result

    def evaluate_into(self, value: ArrayLike, out: FloatArray) -> None:
        np.subtract(self.threshold, _finite_values(value, "value"), out=out)
        out /= self.scale
        np.maximum(out, 0.0, out=out)
        np.power(out, self.curvature, out=out)
        np.negative(out, out=out)

    def marginal_utility(self, value: ArrayLike) -> FloatArray:
        values = np.asarray(value, dtype=float)
        result = np.empty_like(values, dtype=np.float64)
        self.marginal_utility_into(values, result)
        return result

    def marginal_utility_into(self, value: ArrayLike, out: FloatArray) -> None:
        np.subtract(self.threshold, _finite_values(value, "value"), out=out)
        out /= self.scale
        np.maximum(out, 0.0, out=out)
        if self.curvature == 1.0:
            np.greater(out, 0.0, out=out)
        else:
            np.power(out, self.curvature - 1.0, out=out)
        out *= self.curvature / self.scale

    def breakpoints(self) -> tuple[float, ...]:
        return (self.threshold,)

    def diagnostic_breach(self, value: ArrayLike) -> NDArray[np.bool_]:
        return _finite_values(value, "value") < self.threshold


def consumption_utility(
    spending: ArrayLike,
    preferences: Preferences,
) -> FloatArray:
    """Evaluate the built-in consumption preferences before lifetime weighting."""
    values = _finite_values(spending, "spending")
    utility = (1.0 - preferences.bequest_strength) * IsoelasticCurve(
        preferences.consumption_reference,
        preferences.consumption_elasticity,
    ).evaluate(values)
    if preferences.spending_floor > 0.0 and preferences.spending_floor_importance > 0.0:
        utility += preferences.spending_floor_importance * SpendingFloorCurve(
            preferences.spending_floor,
            preferences.spending_floor_scale,
        ).evaluate(values)
    return np.asarray(utility, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class TargetCurve:
    """Symmetric preference around a target; no target is a hard constraint."""

    target: float
    tolerance: float
    curvature: float = 2.0

    def __post_init__(self) -> None:
        _finite_number(self.target, "target")
        _finite_number(self.tolerance, "tolerance")
        _finite_number(self.curvature, "curvature")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if self.curvature < 1:
            raise ValueError("curvature must be at least one")

    def evaluate(self, value: ArrayLike) -> FloatArray:
        values = np.asarray(value, dtype=float)
        result = np.empty_like(values, dtype=np.float64)
        self.evaluate_into(values, result)
        return result

    def evaluate_into(self, value: ArrayLike, out: FloatArray) -> None:
        np.subtract(_finite_values(value, "value"), self.target, out=out)
        np.abs(out, out=out)
        out /= self.tolerance
        np.power(out, self.curvature, out=out)
        np.negative(out, out=out)

    def marginal_utility(self, value: ArrayLike) -> FloatArray:
        values = np.asarray(value, dtype=float)
        result = np.empty_like(values, dtype=np.float64)
        self.marginal_utility_into(values, result)
        return result

    def marginal_utility_into(self, value: ArrayLike, out: FloatArray) -> None:
        values = _finite_values(value, "value")
        below_target = values < self.target
        np.subtract(values, self.target, out=out)
        np.abs(out, out=out)
        out /= self.tolerance
        np.power(out, self.curvature - 1.0, out=out)
        out *= self.curvature / self.tolerance
        out[~below_target] *= -1.0

    def breakpoints(self) -> tuple[float, ...]:
        return (self.target,)

    def diagnostic_breach(self, value: ArrayLike) -> NDArray[np.bool_]:
        return np.abs(_finite_values(value, "value") - self.target) > self.tolerance


@dataclass(frozen=True, slots=True)
class UtilityOutcome:
    """Outcomes available to composable utility add-ons.

    Spending is shaped ``(paths, years)``. Exposure contains alive indicators
    for simulations or survival probabilities for deterministic plans.
    """

    spending: FloatArray
    exposure: FloatArray
    ages: tuple[float, ...]
    terminal_wealth: FloatArray
    decisions: Mapping[OutcomeType, float | FloatArray]

    def __post_init__(self) -> None:
        spending = _finite_values(self.spending, "spending")
        exposure = _finite_values(self.exposure, "exposure")
        terminal = np.atleast_1d(_finite_values(self.terminal_wealth, "terminal_wealth"))
        _finite_values(self.ages, "ages")
        if spending.ndim == 1:
            spending = spending[np.newaxis, :]
        if exposure.ndim == 1:
            exposure = exposure[np.newaxis, :]
        if spending.ndim != 2 or exposure.shape != spending.shape:
            raise ValueError("spending and exposure must be aligned path-by-year arrays")
        if spending.shape[1] != len(self.ages):
            raise ValueError("ages must align with spending years")
        if terminal.size not in (1, spending.shape[0]):
            raise ValueError("terminal_wealth must be scalar or one value per path")
        if terminal.size == 1 and spending.shape[0] > 1:
            terminal = np.full(spending.shape[0], float(terminal[0]))
        decisions = {OutcomeType(name): value for name, value in self.decisions.items()}
        for name, value in decisions.items():
            _finite_values(value, f"decision outcome {name!r}")
        object.__setattr__(self, "spending", spending)
        object.__setattr__(self, "exposure", exposure)
        object.__setattr__(self, "terminal_wealth", terminal)
        object.__setattr__(self, "decisions", decisions)

    @property
    def paths(self) -> int:
        return int(self.spending.shape[0])

    @property
    def available_outcomes(self) -> frozenset[OutcomeType]:
        return frozenset(
            {
                OutcomeType.SPENDING,
                OutcomeType.TERMINAL_WEALTH,
                OutcomeType.BEQUEST,
                *self.decisions,
            }
        )

    def values(self, outcome: OutcomeType) -> FloatArray:
        if outcome is OutcomeType.SPENDING:
            return self.spending
        if outcome is OutcomeType.TERMINAL_WEALTH:
            return self.terminal_wealth
        if outcome is OutcomeType.BEQUEST and outcome not in self.decisions:
            return self.terminal_wealth
        if outcome not in self.decisions:
            raise KeyError(f"utility outcome {outcome!r} is unavailable")
        values = np.asarray(self.decisions[outcome], dtype=float)
        if values.ndim == 0:
            return np.full(self.paths, float(values))
        if values.shape not in {(self.paths,), self.spending.shape}:
            raise ValueError(
                f"decision outcome {outcome!r} must be scalar, path-level, or aligned path-by-year"
            )
        return values


@dataclass(frozen=True, slots=True)
class UtilityAddon:
    """Attach a curve to an outcome and give it explicit relative importance."""

    name: str
    outcome: OutcomeType
    curve: UtilityCurve
    importance: float = 1.0
    aggregation: UtilityAggregation = UtilityAggregation.DISCOUNTED_MEAN
    minimum_age: int | None = None
    maximum_age: int | None = None
    age_reference: float | None = None
    age_growth: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "outcome", OutcomeType(self.outcome))
        object.__setattr__(
            self,
            "aggregation",
            UtilityAggregation(self.aggregation),
        )
        if not self.name:
            raise ValueError("name cannot be empty")
        if not self.outcome:
            raise ValueError("outcome cannot be empty")
        _finite_number(self.importance, "importance")
        if self.importance < 0:
            raise ValueError("importance must be nonnegative")
        for name, value in (
            ("minimum_age", self.minimum_age),
            ("maximum_age", self.maximum_age),
        ):
            if value is not None:
                _finite_number(value, name)
        if (
            self.minimum_age is not None
            and self.maximum_age is not None
            and self.minimum_age > self.maximum_age
        ):
            raise ValueError("minimum_age cannot exceed maximum_age")
        if self.age_reference is not None and not np.isfinite(self.age_reference):
            raise ValueError("age_reference must be finite")
        if not np.isfinite(self.age_growth):
            raise ValueError("age_growth must be finite")
        if self.age_growth != 0.0 and self.age_reference is None:
            raise ValueError("age_reference is required when age_growth is nonzero")

    def age_profile(self, ages: ArrayLike) -> FloatArray:
        values = _finite_values(ages, "ages")
        if self.age_reference is None or self.age_growth == 0.0:
            return np.ones_like(values, dtype=np.float64)
        with np.errstate(over="ignore"):
            exponent = self.age_growth * (values - self.age_reference)
        return np.asarray(np.exp(np.clip(exponent, -700.0, 700.0)), dtype=np.float64)

    def score(self, outcome: UtilityOutcome, weights: FloatArray) -> FloatArray:
        if self.outcome not in outcome.available_outcomes:
            raise KeyError(f"utility outcome {self.outcome!r} is unavailable")
        values = outcome.values(self.outcome)
        evaluated = np.asarray(self.curve.evaluate(values), dtype=float)
        if not np.all(np.isfinite(evaluated)):
            raise ValueError(f"utility add-on {self.name!r} score is not representable")
        if evaluated.ndim == 1:
            if evaluated.shape != (outcome.paths,):
                raise ValueError(f"curve {self.name!r} returned an invalid shape")
            with np.errstate(over="ignore", invalid="ignore"):
                result = self.importance * evaluated
            if not np.all(np.isfinite(result)):
                raise ValueError(f"utility add-on {self.name!r} score is not representable")
            return np.asarray(result, dtype=np.float64)
        if evaluated.shape != outcome.spending.shape:
            raise ValueError(f"curve {self.name!r} returned an invalid shape")
        if evaluated.shape[1] == 0:
            return np.zeros(outcome.paths)
        age_mask = np.ones(len(outcome.ages), dtype=bool)
        ages = np.asarray(outcome.ages)
        if self.minimum_age is not None:
            age_mask &= ages >= self.minimum_age
        if self.maximum_age is not None:
            age_mask &= ages <= self.maximum_age
        finite_weights = _finite_values(weights, "utility weights")
        with np.errstate(over="ignore", invalid="ignore"):
            selected_weights = (
                finite_weights
                * age_mask[np.newaxis, :]
                * self.age_profile(ages)[np.newaxis, :]
            )
            if self.aggregation is UtilityAggregation.DISCOUNTED_SUM:
                aggregated = np.sum(selected_weights * evaluated, axis=1)
            elif self.aggregation is UtilityAggregation.WORST:
                aggregated = np.min(
                    np.where(selected_weights > 0, evaluated, np.inf),
                    axis=1,
                )
                aggregated = np.where(np.isfinite(aggregated), aggregated, 0.0)
            elif self.aggregation is UtilityAggregation.LAST:
                active = selected_weights > 0
                reverse_index = np.argmax(active[:, ::-1], axis=1)
                last_index = active.shape[1] - 1 - reverse_index
                aggregated = evaluated[np.arange(outcome.paths), last_index]
                aggregated = np.where(np.any(active, axis=1), aggregated, 0.0)
            else:
                denominator = selected_weights.sum(axis=1)
                aggregated = np.divide(
                    np.sum(selected_weights * evaluated, axis=1),
                    denominator,
                    out=np.zeros(outcome.paths),
                    where=denominator > 0,
                )
            result = self.importance * aggregated
        if not np.all(np.isfinite(result)):
            raise ValueError(f"utility add-on {self.name!r} score is not representable")
        return np.asarray(result, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class UtilityModel:
    """One additive, inspectable objective for every modeled decision."""

    person: Person
    preferences: Preferences
    addons: tuple[UtilityAddon, ...]

    @staticmethod
    def _from_config(config: UtilityAddonConfig) -> UtilityAddon:
        parameters = config.parameters
        if config.curve is UtilityCurveKind.LINEAR:
            curve: UtilityCurve = LinearCurve(
                slope=parameters.get("slope", 1.0),
                intercept=parameters.get("intercept", 0.0),
            )
        elif config.curve is UtilityCurveKind.ISOELASTIC:
            curve = IsoelasticCurve(
                reference=parameters["reference"],
                elasticity=parameters["elasticity"],
                minimum_utility=parameters.get("minimum_utility", -10.0),
            )
        elif config.curve is UtilityCurveKind.SPENDING_FLOOR:
            curve = SpendingFloorCurve(
                threshold=parameters["threshold"],
                scale=parameters["scale"],
                curvature=parameters.get("curvature", 2.0),
            )
        else:
            curve = TargetCurve(
                target=parameters["target"],
                tolerance=parameters["tolerance"],
                curvature=parameters.get("curvature", 2.0),
            )
        return UtilityAddon(
            name=config.name,
            outcome=config.outcome,
            curve=curve,
            importance=config.importance,
            aggregation=config.aggregation,
            minimum_age=config.minimum_age,
            maximum_age=config.maximum_age,
            age_reference=config.age_reference,
            age_growth=config.age_growth,
        )

    @classmethod
    def from_scenario(
        cls,
        scenario: PlanningScenario,
        extra_addons: Sequence[UtilityAddon] = (),
    ) -> UtilityModel:
        preferences = scenario.preferences
        phi = preferences.bequest_strength
        components: list[UtilityAddon] = [
            UtilityAddon(
                name="consumption",
                outcome=OutcomeType.SPENDING,
                curve=IsoelasticCurve(
                    preferences.consumption_reference,
                    preferences.consumption_elasticity,
                ),
                importance=1.0 - phi,
            )
        ]
        if preferences.spending_floor > 0 and preferences.spending_floor_importance > 0:
            components.append(
                UtilityAddon(
                    name="spending_floor",
                    outcome=OutcomeType.SPENDING,
                    curve=SpendingFloorCurve(
                        preferences.spending_floor,
                        preferences.spending_floor_scale,
                    ),
                    importance=preferences.spending_floor_importance,
                )
            )
        if phi > 0:
            components.append(
                UtilityAddon(
                    name="bequest",
                    outcome=OutcomeType.BEQUEST,
                    curve=IsoelasticCurve(
                        max(preferences.fixed_bequest, 1.0),
                        preferences.bequest_flexibility,
                    ),
                    importance=phi,
                )
            )
        components.extend(cls._from_config(config) for config in scenario.utility_addons)
        components.extend(extra_addons)
        names = [component.name for component in components]
        if len(names) != len(set(names)):
            raise ValueError("utility add-on names must be unique")
        return cls(scenario.person, preferences, tuple(components))

    def weights(self, outcome: UtilityOutcome) -> FloatArray:
        ages = np.asarray(outcome.ages)
        offsets = ages - self.person.current_age
        base = (1.0 + self.preferences.time_preference) ** -offsets * vitality(
            ages, self.preferences
        )
        return np.asarray(outcome.exposure * base[np.newaxis, :], dtype=np.float64)

    def decompose(self, outcome: UtilityOutcome) -> dict[str, FloatArray]:
        weights = self.weights(outcome)
        return {
            addon.name: np.asarray(addon.score(outcome, weights), dtype=np.float64)
            for addon in self.addons
            if addon.outcome in outcome.available_outcomes
        }

    def validate_outcomes(
        self,
        available: frozenset[OutcomeType],
        *,
        require_all: bool = True,
    ) -> tuple[UtilityAddon, ...]:
        """Compile active add-ons and optionally reject unavailable outcomes."""
        missing = tuple(addon for addon in self.addons if addon.outcome not in available)
        if require_all and missing:
            names = ", ".join(f"{addon.name} ({addon.outcome})" for addon in missing)
            raise ValueError(f"utility outcomes unavailable to decision problem: {names}")
        return tuple(addon for addon in self.addons if addon.outcome in available)

    def score(self, outcome: UtilityOutcome) -> FloatArray:
        components = self.decompose(outcome)
        if not components:
            return np.zeros(outcome.paths)
        return np.sum(np.stack(tuple(components.values())), axis=0)


@dataclass(frozen=True, slots=True)
class LifetimeUtility:
    person: Person
    preferences: Preferences

    def weights(self, length: int | None = None) -> FloatArray:
        if length is None:
            length = self.person.horizon
        discount = (1.0 + self.preferences.time_preference) ** -np.arange(length)
        ages = np.arange(self.person.current_age, self.person.current_age + length)
        return np.asarray(
            discount * vitality(ages, self.preferences),
            dtype=np.float64,
        )

    def score(
        self,
        consumption: ArrayLike,
        *,
        survival: ArrayLike | None = None,
        bequest: float = 0.0,
        bequest_divisor: float = 1.0,
    ) -> float:
        values = np.asarray(consumption, dtype=float)
        exposure = np.ones(len(values)) if survival is None else np.asarray(survival, dtype=float)
        phi = self.preferences.bequest_strength
        addons = [
            UtilityAddon(
                "consumption",
                OutcomeType.SPENDING,
                IsoelasticCurve(
                    self.preferences.consumption_reference,
                    self.preferences.consumption_elasticity,
                ),
                1.0 - phi,
            )
        ]
        if self.preferences.spending_floor_importance > 0:
            addons.append(
                UtilityAddon(
                    "spending_floor",
                    OutcomeType.SPENDING,
                    SpendingFloorCurve(
                        self.preferences.spending_floor,
                        self.preferences.spending_floor_scale,
                    ),
                    self.preferences.spending_floor_importance,
                )
            )
        if phi > 0:
            _finite_number(bequest_divisor, "bequest_divisor")
            if bequest_divisor <= 0:
                raise ValueError("bequest_divisor must be positive")
            addons.append(
                UtilityAddon(
                    "bequest",
                    OutcomeType.BEQUEST,
                    IsoelasticCurve(
                        self.preferences.consumption_reference * bequest_divisor,
                        self.preferences.bequest_flexibility,
                    ),
                    phi,
                )
            )
        outcome = UtilityOutcome(
            spending=values,
            exposure=exposure,
            ages=tuple(range(self.person.current_age, self.person.current_age + len(values))),
            terminal_wealth=np.array([bequest]),
            decisions={OutcomeType.BEQUEST: bequest},
        )
        model = UtilityModel(self.person, self.preferences, tuple(addons))
        return float(model.score(outcome)[0])

    def certainty_equivalent(
        self,
        consumption: ArrayLike,
        *,
        survival: ArrayLike | None = None,
    ) -> float:
        values = np.asarray(consumption, dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("consumption must be finite and nonnegative")
        weights = self.weights(len(values))
        if survival is not None:
            probabilities = np.asarray(survival, dtype=float)
            if probabilities.shape != values.shape:
                raise ValueError("survival must align with consumption")
            if (
                not np.all(np.isfinite(probabilities))
                or np.any(probabilities < 0)
                or np.any(probabilities > 1)
            ):
                raise ValueError("survival must contain finite probabilities between zero and one")
            weights *= probabilities
        return float(
            _weighted_certainty_equivalent(
                values,
                weights,
                self.preferences.consumption_elasticity,
            )
        )


def bequest_divisor(
    survival: ArrayLike,
    time_preference: float,
) -> float:
    probabilities = np.asarray(survival, dtype=float)
    years = np.arange(len(probabilities))
    return float(np.sum(probabilities / (1.0 + time_preference) ** years))
