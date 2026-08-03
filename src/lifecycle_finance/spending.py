"""Composable vectorized spending policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from .domain import (
    LifecyclePlan,
    OutcomeType,
    PlanningScenario,
    UtilityAggregation,
)
from .utility import (
    DifferentiableUtilityCurve,
    UtilityAddon,
    UtilityCurve,
    UtilityModel,
    vitality,
)

FloatArray = NDArray[np.float64]


def _discounted_annuity_and_income(
    real_rate: FloatArray,
    conditional_survival: FloatArray,
    future_income: FloatArray,
    current_income: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    discount = 1.0 / (1.0 + real_rate)
    annuity_coefficients = conditional_survival.copy()
    annuity_coefficients[0] = 0.0
    income_coefficients = conditional_survival * future_income
    income_coefficients[0] = 0.0
    annuity = np.polynomial.polynomial.polyval(discount, annuity_coefficients)
    income = current_income + np.polynomial.polynomial.polyval(
        discount,
        income_coefficients,
    )
    return (
        np.asarray(annuity, dtype=np.float64),
        np.asarray(income, dtype=np.float64),
    )


def _method_owner(instance: object, name: str) -> type[object] | None:
    return next(
        (owner for owner in type(instance).__mro__ if name in owner.__dict__),
        None,
    )


def _evaluate_marginal_utility(
    curve: DifferentiableUtilityCurve,
    values: FloatArray,
    out: FloatArray,
) -> None:
    """Evaluate a derivative into ``out``, preserving custom curve support."""
    evaluate_into = getattr(curve, "marginal_utility_into", None)
    if callable(evaluate_into) and _method_owner(
        curve,
        "marginal_utility",
    ) is _method_owner(curve, "marginal_utility_into"):
        evaluate_into(values, out)
    else:
        out[:] = curve.marginal_utility(values)


def _evaluate_utility(
    curve: UtilityCurve,
    values: FloatArray,
    out: FloatArray,
) -> None:
    """Evaluate a curve into ``out``, preserving custom curve support."""
    evaluate_into = getattr(curve, "evaluate_into", None)
    if callable(evaluate_into) and _method_owner(curve, "evaluate") is _method_owner(
        curve,
        "evaluate_into",
    ):
        evaluate_into(values, out)
    else:
        out[:] = curve.evaluate(values)


@dataclass(frozen=True, slots=True)
class SpendingContext:
    year: int
    wealth: FloatArray
    income: FloatArray
    future_income: FloatArray
    real_rate: FloatArray
    scenario: PlanningScenario
    lifecycle_plan: LifecyclePlan

    @property
    def age(self) -> int:
        return self.scenario.person.current_age + self.year


class SpendingPolicy(Protocol):
    def target(self, context: SpendingContext) -> FloatArray:
        """Return desired total spending for every simulated path."""


@dataclass(frozen=True, slots=True)
class SpendingOptimizationProblem:
    resources: FloatArray
    future_annuity: FloatArray
    addons: tuple[UtilityAddon, ...]
    current_weights: FloatArray
    future_weights: FloatArray
    aggregations: tuple[UtilityAggregation, ...]

    def __post_init__(self) -> None:
        resources = np.asarray(self.resources, dtype=float)
        future_annuity = np.asarray(self.future_annuity, dtype=float)
        current_weights = np.asarray(self.current_weights, dtype=float)
        future_weights = np.asarray(self.future_weights, dtype=float)
        addon_count = len(self.addons)
        if resources.ndim != 1 or future_annuity.shape != resources.shape:
            raise ValueError("resources and future_annuity must be aligned vectors")
        if current_weights.shape != (addon_count,) or future_weights.shape != (addon_count,):
            raise ValueError("utility weights must contain one value per add-on")
        if len(self.aggregations) != addon_count:
            raise ValueError("aggregations must contain one value per add-on")
        if np.any(resources < 0) or np.any(future_annuity < 0):
            raise ValueError("resources and future_annuity cannot be negative")
        if np.any(current_weights < 0) or np.any(future_weights < 0):
            raise ValueError("utility weights cannot be negative")
        object.__setattr__(self, "resources", resources)
        object.__setattr__(self, "future_annuity", future_annuity)
        object.__setattr__(self, "current_weights", current_weights)
        object.__setattr__(self, "future_weights", future_weights)
        object.__setattr__(
            self,
            "aggregations",
            tuple(UtilityAggregation(value) for value in self.aggregations),
        )


@dataclass(frozen=True, slots=True)
class SpendingSolution:
    spending: FloatArray
    score: FloatArray


def build_spending_problem(
    *,
    utility_model: UtilityModel,
    resources: FloatArray,
    future_annuity: FloatArray,
    ages: FloatArray,
    conditional_survival: FloatArray,
) -> SpendingOptimizationProblem:
    """Build the spending-only rolling utility problem for a shared age window."""
    offsets = np.arange(len(ages))
    utility_weights = (
        (1.0 + utility_model.preferences.time_preference) ** -offsets
        * conditional_survival
        * vitality(ages, utility_model.preferences)
    )
    addons: list[UtilityAddon] = []
    current_weights: list[float] = []
    future_weights: list[float] = []
    for addon in utility_model.addons:
        if addon.outcome is not OutcomeType.SPENDING:
            continue
        age_mask = np.ones(len(ages), dtype=bool)
        if addon.minimum_age is not None:
            age_mask &= ages >= addon.minimum_age
        if addon.maximum_age is not None:
            age_mask &= ages <= addon.maximum_age
        weights = utility_weights * age_mask * addon.age_profile(ages)
        current_weight = float(weights[0])
        future_weight = float(np.sum(weights[1:]))
        if current_weight + future_weight <= 0:
            continue
        addons.append(addon)
        current_weights.append(current_weight)
        future_weights.append(future_weight)
    return SpendingOptimizationProblem(
        resources=resources,
        future_annuity=future_annuity,
        addons=tuple(addons),
        current_weights=np.asarray(current_weights, dtype=float),
        future_weights=np.asarray(future_weights, dtype=float),
        aggregations=tuple(addon.aggregation for addon in addons),
    )


class SpendingSolver(Protocol):
    def solve(self, problem: SpendingOptimizationProblem) -> FloatArray:
        """Return utility-maximizing current spending by path."""

    def objective(
        self,
        problem: SpendingOptimizationProblem,
        candidates: FloatArray,
    ) -> FloatArray:
        """Evaluate candidate current spending in utility points."""


@dataclass(frozen=True, slots=True)
class DerivativeSpendingSolver:
    """Piecewise derivative solver with a grid fallback for custom curves."""

    iterations: int = 16
    fallback_grid_size: int = 41

    def __post_init__(self) -> None:
        if self.iterations < 8:
            raise ValueError("iterations must be at least 8")
        if self.fallback_grid_size < 5:
            raise ValueError("fallback_grid_size must be at least 5")

    @staticmethod
    def objective(
        problem: SpendingOptimizationProblem,
        candidates: FloatArray,
    ) -> FloatArray:
        annuity = problem.future_annuity[:, np.newaxis]
        future = np.divide(
            problem.resources[:, np.newaxis] - candidates,
            annuity,
            out=np.zeros_like(candidates),
            where=annuity > 0,
        )
        scores = np.zeros_like(candidates)
        current_utility = np.empty_like(candidates)
        future_utility = np.empty_like(candidates)
        component = np.empty_like(candidates)
        term = np.empty_like(candidates)
        for index, addon in enumerate(problem.addons):
            current_weight = problem.current_weights[index]
            future_weight = problem.future_weights[index]
            denominator = current_weight + future_weight
            if denominator <= 0:
                continue
            _evaluate_utility(addon.curve, candidates, current_utility)
            _evaluate_utility(addon.curve, future, future_utility)
            aggregation = problem.aggregations[index]
            if aggregation is UtilityAggregation.WORST:
                if current_weight <= 0:
                    component[:] = future_utility
                elif future_weight <= 0:
                    component[:] = current_utility
                else:
                    np.minimum(current_utility, future_utility, out=component)
            elif aggregation is UtilityAggregation.LAST:
                component[:] = future_utility if future_weight > 0 else current_utility
            else:
                normalizer = (
                    1.0 if aggregation is UtilityAggregation.DISCOUNTED_SUM else denominator
                )
                np.multiply(current_utility, current_weight, out=component)
                np.multiply(future_utility, future_weight, out=term)
                component += term
                component /= normalizer
            component *= addon.importance
            scores += component
        return scores

    def _grid(
        self,
        problem: SpendingOptimizationProblem,
        maximum_spending: FloatArray,
    ) -> SpendingSolution:
        fractions = np.linspace(0.0, 1.0, self.fallback_grid_size)
        candidates = maximum_spending[:, np.newaxis] * fractions[np.newaxis, :]
        scores = self.objective(problem, candidates)
        optimum = np.argmax(scores, axis=1)
        path_indices = np.arange(len(problem.resources))
        return SpendingSolution(
            spending=np.asarray(candidates[path_indices, optimum], dtype=np.float64),
            score=np.asarray(scores[path_indices, optimum], dtype=np.float64),
        )

    @staticmethod
    def _is_differentiable(problem: SpendingOptimizationProblem) -> bool:
        return all(
            aggregation is not UtilityAggregation.WORST
            and callable(getattr(addon.curve, "marginal_utility", None))
            and callable(getattr(addon.curve, "breakpoints", None))
            for addon, aggregation in zip(
                problem.addons,
                problem.aggregations,
                strict=True,
            )
        )

    @staticmethod
    def _maximum_spending(
        problem: SpendingOptimizationProblem,
        maximum_spending: FloatArray | None,
    ) -> FloatArray:
        if maximum_spending is None:
            return problem.resources.copy()
        maximum = np.asarray(maximum_spending, dtype=float)
        try:
            maximum = np.broadcast_to(maximum, problem.resources.shape)
        except ValueError as error:
            raise ValueError("maximum_spending must align with resources") from error
        if not np.all(np.isfinite(maximum)):
            raise ValueError("maximum_spending must contain only finite values")
        if np.any(maximum < 0.0):
            raise ValueError("maximum_spending cannot be negative")
        return np.asarray(np.minimum(maximum, problem.resources), dtype=np.float64)

    @staticmethod
    def _marginal_difference(
        problem: SpendingOptimizationProblem,
        current: FloatArray,
        *,
        future: FloatArray | None = None,
        difference: FloatArray | None = None,
        current_marginal: FloatArray | None = None,
        future_marginal: FloatArray | None = None,
        term: FloatArray | None = None,
    ) -> FloatArray:
        if future is None:
            future = np.empty_like(current)
        future.fill(0.0)
        np.divide(
            problem.resources - current,
            problem.future_annuity,
            out=future,
            where=problem.future_annuity > 0,
        )
        if difference is None:
            difference = np.empty_like(current)
        if current_marginal is None:
            current_marginal = np.empty_like(current)
        if future_marginal is None:
            future_marginal = np.empty_like(current)
        if term is None:
            term = np.empty_like(current)
        difference.fill(0.0)
        for index, addon in enumerate(problem.addons):
            current_weight = problem.current_weights[index]
            future_weight = problem.future_weights[index]
            denominator = current_weight + future_weight
            if denominator <= 0:
                continue
            curve = cast(DifferentiableUtilityCurve, addon.curve)
            _evaluate_marginal_utility(curve, current, current_marginal)
            _evaluate_marginal_utility(curve, future, future_marginal)
            aggregation = problem.aggregations[index]
            if aggregation is UtilityAggregation.LAST:
                if future_weight > 0:
                    term.fill(0.0)
                    np.divide(
                        future_marginal,
                        problem.future_annuity,
                        out=term,
                        where=problem.future_annuity > 0,
                    )
                    difference -= addon.importance * term
                else:
                    difference += addon.importance * current_marginal
                continue
            normalizer = 1.0 if aggregation is UtilityAggregation.DISCOUNTED_SUM else denominator
            term.fill(0.0)
            np.divide(
                future_marginal,
                problem.future_annuity,
                out=term,
                where=problem.future_annuity > 0,
            )
            term *= future_weight
            current_marginal *= current_weight
            current_marginal -= term
            current_marginal *= addon.importance / normalizer
            difference += current_marginal
        return difference

    def solve_with_score(
        self,
        problem: SpendingOptimizationProblem,
        maximum_spending: FloatArray | None = None,
    ) -> SpendingSolution:
        maximum = self._maximum_spending(problem, maximum_spending)
        if not problem.addons:
            return SpendingSolution(
                spending=np.zeros_like(problem.resources),
                score=np.zeros_like(problem.resources),
            )
        if not self._is_differentiable(problem):
            return self._grid(problem, maximum)

        lower = np.zeros_like(problem.resources)
        upper = maximum.copy()
        midpoint = np.empty_like(problem.resources)
        future = np.empty_like(problem.resources)
        difference = np.empty_like(problem.resources)
        current_marginal = np.empty_like(problem.resources)
        future_marginal = np.empty_like(problem.resources)
        term = np.empty_like(problem.resources)
        for _ in range(self.iterations):
            np.add(lower, upper, out=midpoint)
            midpoint *= 0.5
            derivative = self._marginal_difference(
                problem,
                midpoint,
                future=future,
                difference=difference,
                current_marginal=current_marginal,
                future_marginal=future_marginal,
                term=term,
            )
            increasing = derivative > 0.0
            np.copyto(lower, midpoint, where=increasing)
            np.copyto(upper, midpoint, where=~increasing)
        root = (lower + upper) / 2.0

        candidates = [
            np.zeros_like(problem.resources),
            maximum,
            root,
        ]
        seen_breakpoints: set[float] = set()
        unique_breakpoints: list[float] = []
        for addon in problem.addons:
            breakpoints = cast(
                DifferentiableUtilityCurve,
                addon.curve,
            ).breakpoints()
            for breakpoint in breakpoints:
                numeric_breakpoint = float(breakpoint)
                if numeric_breakpoint in seen_breakpoints:
                    continue
                seen_breakpoints.add(numeric_breakpoint)
                unique_breakpoints.append(numeric_breakpoint)
        for breakpoint in unique_breakpoints:
            candidates.append(np.minimum(maximum, max(breakpoint, 0.0)))
            candidates.append(
                np.clip(
                    problem.resources - problem.future_annuity * breakpoint,
                    0.0,
                    maximum,
                )
            )
        candidate_matrix = np.stack(candidates, axis=1)
        scores = self.objective(problem, candidate_matrix)
        optimum = np.argmax(scores, axis=1)
        path_indices = np.arange(len(problem.resources))
        return SpendingSolution(
            spending=np.asarray(candidate_matrix[path_indices, optimum], dtype=np.float64),
            score=np.asarray(scores[path_indices, optimum], dtype=np.float64),
        )

    def solve(self, problem: SpendingOptimizationProblem) -> FloatArray:
        return self.solve_with_score(problem).spending


@dataclass(frozen=True, slots=True)
class PlanSpending:
    """Follow deterministic lifecycle consumption unless resources are insufficient."""

    include_nondiscretionary: bool = True

    def target(self, context: SpendingContext) -> FloatArray:
        discretionary = context.lifecycle_plan.discretionary_consumption_path[context.year]
        nondiscretionary = (
            context.scenario.preferences.nondiscretionary_consumption
            if self.include_nondiscretionary
            else 0.0
        )
        return np.full_like(context.wealth, discretionary + nondiscretionary)


@dataclass(frozen=True, slots=True)
class FixedSpending:
    annual_amount: float
    real_growth: float = 0.0

    def __post_init__(self) -> None:
        if not isfinite(self.annual_amount):
            raise ValueError("annual_amount must be finite")
        if self.annual_amount < 0:
            raise ValueError("annual_amount cannot be negative")
        if not isfinite(self.real_growth):
            raise ValueError("real_growth must be finite")

    def target(self, context: SpendingContext) -> FloatArray:
        amount = self.annual_amount * (1.0 + self.real_growth) ** context.year
        return np.full_like(context.wealth, amount)


@dataclass(frozen=True, slots=True)
class AmortizedSpending:
    def target(self, context: SpendingContext) -> FloatArray:
        survival = np.asarray(
            context.lifecycle_plan.survival_probabilities[context.year :],
            dtype=float,
        )
        conditional = survival / max(survival[0], np.finfo(float).tiny)
        future_annuity, income_pv = _discounted_annuity_and_income(
            np.maximum(context.real_rate, -0.99),
            conditional,
            context.future_income[context.year :],
            context.income,
        )
        resources = context.wealth + income_pv
        return np.asarray(
            resources / np.maximum(1.0 + future_annuity, 1.0),
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class MarginalUtilitySpending:
    """Closed-form CRRA spending without hard preference constraints."""

    expected_return_spread: float = 0.0

    def target(self, context: SpendingContext) -> FloatArray:
        preferences = context.scenario.preferences
        horizon = len(context.future_income)
        remaining = horizon - context.year
        offsets = np.arange(remaining)
        ages = np.arange(context.age, context.age + remaining)

        financial_rate = np.maximum(
            context.real_rate + self.expected_return_spread,
            -0.99,
        )
        discount = np.power(1.0 + financial_rate[:, None], -offsets[None, :])
        beta = 1.0 / (1.0 + preferences.time_preference)

        survival = np.asarray(
            context.lifecycle_plan.survival_probabilities[context.year :],
            dtype=float,
        )
        conditional = survival / max(survival[0], np.finfo(float).tiny)
        utility_weight = np.power(beta, offsets) * conditional * vitality(ages, preferences)

        elasticity = preferences.consumption_elasticity
        log_phi = elasticity * (
            np.log(np.maximum(utility_weight, np.finfo(float).tiny))[None, :]
            - np.log(np.maximum(discount, np.finfo(float).tiny))
        )
        log_phi -= np.max(log_phi, axis=1, keepdims=True)
        phi = np.exp(log_phi)
        weighted_annuity = np.sum(phi * discount, axis=1)

        future_income = np.broadcast_to(
            context.future_income[context.year :],
            (len(context.wealth), remaining),
        ).copy()
        future_income[:, 0] = context.income
        income_pv = np.sum(
            discount * conditional[None, :] * future_income,
            axis=1,
        )
        resources = context.wealth + income_pv
        return np.asarray(
            resources
            * phi[:, 0]
            / np.maximum(weighted_annuity, np.finfo(float).tiny),
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class UtilityOptimizedSpending:
    """Choose current spending from the same additive utility used for decisions.

    The rolling-horizon approximation compares current spending with a
    level future-spending continuation. Built-in curves use analytic
    marginal utility; custom curves fall back to a bounded grid.
    """

    utility_model: UtilityModel
    expected_return_spread: float = 0.0
    solver: SpendingSolver = field(default_factory=DerivativeSpendingSolver)

    def target(self, context: SpendingContext) -> FloatArray:
        horizon = len(context.future_income)
        remaining = horizon - context.year
        ages = np.arange(context.age, context.age + remaining)
        financial_rate = np.maximum(
            context.real_rate + self.expected_return_spread,
            -0.99,
        )
        survival = np.asarray(
            context.lifecycle_plan.survival_probabilities[context.year :],
            dtype=float,
        )
        conditional = survival / max(survival[0], np.finfo(float).tiny)
        future_annuity, income_pv = _discounted_annuity_and_income(
            financial_rate,
            conditional,
            context.future_income[context.year :],
            context.income,
        )
        resources = context.wealth + income_pv
        resources = np.maximum(resources, 0.0)

        return self.solver.solve(
            build_spending_problem(
                utility_model=self.utility_model,
                resources=resources,
                future_annuity=future_annuity,
                ages=ages.astype(float),
                conditional_survival=conditional,
            )
        )
