"""Vectorized income-risk models for stochastic lifecycle projections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int_]


def _float_vector(name: str, values: object) -> FloatArray:
    array = np.array(values, dtype=float, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    array.setflags(write=False)
    return array


def _bool_vector(name: str, values: object) -> BoolArray:
    original = np.asarray(values)
    if original.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.issubdtype(original.dtype, np.bool_):
        raise ValueError(f"{name} must be boolean")
    array = np.array(original, dtype=bool, copy=True)
    array.setflags(write=False)
    return array


def _float_matrix(name: str, values: object) -> FloatArray:
    array = np.array(values, dtype=float, copy=True)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    array.setflags(write=False)
    return array


def _bool_matrix(name: str, values: object) -> BoolArray:
    original = np.asarray(values)
    if original.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if not np.issubdtype(original.dtype, np.bool_):
        raise ValueError(f"{name} must be boolean")
    array = np.array(original, dtype=bool, copy=True)
    array.setflags(write=False)
    return array


def _fraction(name: str, value: float) -> None:
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")


@dataclass(frozen=True, slots=True)
class IncomeRiskContext:
    """Path-level inputs for one model year."""

    year: int
    deterministic_income: FloatArray
    working: BoolArray
    current_excess_return: FloatArray
    lagged_excess_return: FloatArray
    random_uniform: FloatArray

    def __post_init__(self) -> None:
        if isinstance(self.year, bool) or not isinstance(self.year, (int, np.integer)):
            raise ValueError("year must be a nonnegative integer")
        if self.year < 0:
            raise ValueError("year must be a nonnegative integer")
        deterministic_income = _float_vector("deterministic_income", self.deterministic_income)
        working = _bool_vector("working", self.working)
        current_excess_return = _float_vector(
            "current_excess_return", self.current_excess_return
        )
        lagged_excess_return = _float_vector("lagged_excess_return", self.lagged_excess_return)
        random_uniform = _float_vector("random_uniform", self.random_uniform)
        path_count = deterministic_income.shape
        for name, values in (
            ("working", working),
            ("current_excess_return", current_excess_return),
            ("lagged_excess_return", lagged_excess_return),
            ("random_uniform", random_uniform),
        ):
            if values.shape != path_count:
                raise ValueError(f"{name} must match deterministic_income shape")
        if np.any((random_uniform < 0.0) | (random_uniform > 1.0)):
            raise ValueError("random_uniform must be in [0, 1]")
        object.__setattr__(self, "deterministic_income", deterministic_income)
        object.__setattr__(self, "working", working)
        object.__setattr__(self, "current_excess_return", current_excess_return)
        object.__setattr__(self, "lagged_excess_return", lagged_excess_return)
        object.__setattr__(self, "random_uniform", random_uniform)


@dataclass(frozen=True, slots=True)
class IncomeRiskState:
    """Persistent path state retained by an income-risk model."""

    displaced: BoolArray
    years_since_displacement: IntArray

    def __post_init__(self) -> None:
        displaced = _bool_vector("displaced", self.displaced)
        original_years = np.asarray(self.years_since_displacement)
        if original_years.ndim != 1 or not np.issubdtype(original_years.dtype, np.integer):
            raise ValueError("years_since_displacement must be a one-dimensional integer array")
        years_since_displacement = np.array(original_years, dtype=int, copy=True)
        if np.any(years_since_displacement < 0):
            raise ValueError("years_since_displacement must be nonnegative")
        if displaced.shape != years_since_displacement.shape:
            raise ValueError("years_since_displacement must match displaced shape")
        years_since_displacement.setflags(write=False)
        object.__setattr__(self, "displaced", displaced)
        object.__setattr__(self, "years_since_displacement", years_since_displacement)


@dataclass(frozen=True, slots=True)
class IncomeRiskStep:
    """A model transition's realized income, state, and vesting result."""

    state: IncomeRiskState
    realized_income: FloatArray
    income_fraction: FloatArray
    vesting_eligible: BoolArray

    def __post_init__(self) -> None:
        realized_income = _float_vector("realized_income", self.realized_income)
        income_fraction = _float_vector("income_fraction", self.income_fraction)
        vesting_eligible = _bool_vector("vesting_eligible", self.vesting_eligible)
        shape = self.state.displaced.shape
        for name, values in (
            ("realized_income", realized_income),
            ("income_fraction", income_fraction),
            ("vesting_eligible", vesting_eligible),
        ):
            if values.shape != shape:
                raise ValueError(f"{name} must match state shape")
        if np.any((income_fraction < 0.0) | (income_fraction > 1.0)):
            raise ValueError("income_fraction must be in [0, 1]")
        object.__setattr__(self, "realized_income", realized_income)
        object.__setattr__(self, "income_fraction", income_fraction)
        object.__setattr__(self, "vesting_eligible", vesting_eligible)


@dataclass(frozen=True, slots=True)
class IncomeRiskPaths:
    """Path-by-year income-risk results."""

    realized_income: FloatArray
    income_fraction: FloatArray
    displaced: BoolArray
    vesting_eligible: BoolArray

    def __post_init__(self) -> None:
        realized_income = _float_matrix("realized_income", self.realized_income)
        income_fraction = _float_matrix("income_fraction", self.income_fraction)
        displaced = _bool_matrix("displaced", self.displaced)
        vesting_eligible = _bool_matrix("vesting_eligible", self.vesting_eligible)
        shape = realized_income.shape
        for name, values in (
            ("income_fraction", income_fraction),
            ("displaced", displaced),
            ("vesting_eligible", vesting_eligible),
        ):
            if values.shape != shape:
                raise ValueError(f"{name} must match realized_income shape")
        if np.any((income_fraction < 0.0) | (income_fraction > 1.0)):
            raise ValueError("income_fraction must be in [0, 1]")
        object.__setattr__(self, "realized_income", realized_income)
        object.__setattr__(self, "income_fraction", income_fraction)
        object.__setattr__(self, "displaced", displaced)
        object.__setattr__(self, "vesting_eligible", vesting_eligible)


class IncomeRiskModel(Protocol):
    def initial_state(self, paths: int) -> IncomeRiskState: ...

    def transition(
        self,
        context: IncomeRiskContext,
        state: IncomeRiskState,
    ) -> IncomeRiskStep: ...


def _initial_state(paths: int) -> IncomeRiskState:
    if isinstance(paths, bool) or not isinstance(paths, (int, np.integer)) or paths < 0:
        raise ValueError("paths must be a nonnegative integer")
    return IncomeRiskState(
        displaced=np.zeros(paths, dtype=bool),
        years_since_displacement=np.zeros(paths, dtype=int),
    )


def _validate_state_shape(context: IncomeRiskContext, state: IncomeRiskState) -> None:
    if state.displaced.shape != context.deterministic_income.shape:
        raise ValueError("state must match context path shape")


def _validate_income_risk_state_paths(
    state: IncomeRiskState,
    expected_paths: int,
) -> None:
    if state.displaced.shape != (expected_paths,):
        raise ValueError(
            f"income risk initial state must contain exactly {expected_paths} paths"
        )


def _validate_income_risk_step_paths(
    step: IncomeRiskStep,
    expected_paths: int,
) -> None:
    expected_shape = (expected_paths,)
    if any(
        values.shape != expected_shape
        for values in (
            step.state.displaced,
            step.realized_income,
            step.income_fraction,
            step.vesting_eligible,
        )
    ):
        raise ValueError(
            f"income risk transition step must contain exactly {expected_paths} paths"
        )


@dataclass(frozen=True, slots=True)
class TransitoryMarketJobLoss:
    """One-year job losses whose hazard rises with the current market draw."""

    baseline_probability: float
    market_sensitivity: float
    income_fraction: float
    probability_cap: float

    def __post_init__(self) -> None:
        _fraction("baseline_probability", self.baseline_probability)
        if not np.isfinite(self.market_sensitivity):
            raise ValueError("market_sensitivity must be finite")
        _fraction("income_fraction", self.income_fraction)
        _fraction("probability_cap", self.probability_cap)

    def initial_state(self, paths: int) -> IncomeRiskState:
        return _initial_state(paths)

    def transition(
        self,
        context: IncomeRiskContext,
        state: IncomeRiskState,
    ) -> IncomeRiskStep:
        _validate_state_shape(context, state)
        with np.errstate(over="ignore", invalid="ignore"):
            probability = np.minimum(
                self.baseline_probability
                * np.exp(
                    self.market_sensitivity * np.maximum(-context.current_excess_return, 0.0)
                ),
                self.probability_cap,
            )
        if self.baseline_probability == 0.0:
            probability = np.zeros_like(context.current_excess_return)
        lost = context.working & (context.random_uniform < probability)
        income_fraction = np.where(lost, self.income_fraction, 1.0)
        realized_income = context.deterministic_income * income_fraction
        return IncomeRiskStep(
            state=state,
            realized_income=realized_income,
            income_fraction=income_fraction,
            vesting_eligible=np.ones_like(context.working),
        )


@dataclass(frozen=True, slots=True)
class PersistentDisplacementIncomeRisk:
    """Market-sensitive displacement with a working-year recovery schedule."""

    baseline_probability: float
    market_sensitivity: float
    probability_cap: float
    income_fractions_after_displacement: tuple[float, ...]

    def __post_init__(self) -> None:
        _fraction("baseline_probability", self.baseline_probability)
        if not np.isfinite(self.market_sensitivity):
            raise ValueError("market_sensitivity must be finite")
        _fraction("probability_cap", self.probability_cap)
        if not self.income_fractions_after_displacement:
            raise ValueError("income_fractions_after_displacement must not be empty")
        fractions = tuple(float(value) for value in self.income_fractions_after_displacement)
        for index, fraction in enumerate(fractions):
            _fraction(f"income_fractions_after_displacement[{index}]", fraction)
        object.__setattr__(self, "income_fractions_after_displacement", fractions)

    def initial_state(self, paths: int) -> IncomeRiskState:
        return _initial_state(paths)

    def transition(
        self,
        context: IncomeRiskContext,
        state: IncomeRiskState,
    ) -> IncomeRiskStep:
        _validate_state_shape(context, state)
        excess_return = (
            np.zeros_like(context.lagged_excess_return)
            if context.year == 0
            else context.lagged_excess_return
        )
        with np.errstate(over="ignore", invalid="ignore"):
            probability = np.minimum(
                self.baseline_probability
                * np.exp(self.market_sensitivity * np.maximum(-excess_return, 0.0)),
                self.probability_cap,
            )
        if self.baseline_probability == 0.0:
            probability = np.zeros_like(excess_return)
        newly_displaced = (
            context.working & ~state.displaced & (context.random_uniform < probability)
        )
        displaced = state.displaced | newly_displaced
        years_since_displacement = np.where(
            state.displaced & context.working,
            state.years_since_displacement + 1,
            state.years_since_displacement,
        )
        years_since_displacement = np.where(newly_displaced, 0, years_since_displacement)
        recovery_index = np.minimum(
            years_since_displacement,
            len(self.income_fractions_after_displacement) - 1,
        )
        recovery_fraction = np.asarray(self.income_fractions_after_displacement)[recovery_index]
        income_fraction = np.where(displaced & context.working, recovery_fraction, 1.0)
        return IncomeRiskStep(
            state=IncomeRiskState(
                displaced=displaced,
                years_since_displacement=years_since_displacement,
            ),
            realized_income=context.deterministic_income * income_fraction,
            income_fraction=income_fraction,
            vesting_eligible=~displaced,
        )


def generate_income_risk_paths(
    model: IncomeRiskModel,
    *,
    deterministic_income: FloatArray,
    equity_returns: FloatArray,
    real_rates: FloatArray,
    equity_risk_premium: float,
    working_years: int,
    random_uniforms: FloatArray,
) -> IncomeRiskPaths:
    """Generate income-risk outcomes with a single loop over modeled years."""
    deterministic_income = _float_vector("deterministic_income", deterministic_income)
    equity_returns = _float_matrix("equity_returns", equity_returns)
    real_rates = _float_matrix("real_rates", real_rates)
    random_uniforms = _float_matrix("random_uniforms", random_uniforms)
    if not np.isfinite(equity_risk_premium):
        raise ValueError("equity_risk_premium must be finite")
    if (
        isinstance(working_years, bool)
        or not isinstance(working_years, (int, np.integer))
        or working_years < 0
    ):
        raise ValueError("working_years must be a nonnegative integer")
    paths, horizon = equity_returns.shape
    if deterministic_income.shape != (horizon,):
        raise ValueError("deterministic_income must have one value per model year")
    if real_rates.shape != (paths, horizon):
        raise ValueError("real_rates must match equity_returns shape")
    if random_uniforms.shape != (paths, horizon):
        raise ValueError("random_uniforms must match equity_returns shape")
    if np.any((random_uniforms < 0.0) | (random_uniforms > 1.0)):
        raise ValueError("random_uniforms must be in [0, 1]")

    realized_income = np.empty((paths, horizon), dtype=float)
    income_fraction = np.empty((paths, horizon), dtype=float)
    displaced = np.empty((paths, horizon), dtype=bool)
    vesting_eligible = np.empty((paths, horizon), dtype=bool)
    excess_returns = equity_returns - real_rates - equity_risk_premium
    state = model.initial_state(paths)
    _validate_income_risk_state_paths(state, paths)

    for year in range(horizon):
        step = model.transition(
            IncomeRiskContext(
                year=year,
                deterministic_income=np.full(paths, deterministic_income[year]),
                working=np.full(paths, year < working_years, dtype=bool),
                current_excess_return=excess_returns[:, year],
                lagged_excess_return=(
                    np.zeros(paths, dtype=float)
                    if year == 0
                    else excess_returns[:, year - 1]
                ),
                random_uniform=random_uniforms[:, year],
            ),
            state,
        )
        _validate_income_risk_step_paths(step, paths)
        state = step.state
        realized_income[:, year] = step.realized_income
        income_fraction[:, year] = step.income_fraction
        displaced[:, year] = step.state.displaced
        vesting_eligible[:, year] = step.vesting_eligible

    return IncomeRiskPaths(
        realized_income=realized_income,
        income_fraction=income_fraction,
        displaced=displaced,
        vesting_eligible=vesting_eligible,
    )
