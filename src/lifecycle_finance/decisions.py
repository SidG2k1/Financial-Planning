"""Joint rolling decisions under one composite utility objective."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .domain import (
    Allocation,
    MarketModelConfig,
    OutcomeType,
    UtilityAggregation,
)
from .spending import (
    DerivativeSpendingSolver,
    SpendingContext,
    SpendingSolver,
    _discounted_annuity_and_income,
    build_spending_problem,
)
from .utility import UtilityAddon, UtilityCurve, UtilityModel, vitality

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def optimize_resource_split(
    resources: ArrayLike,
    retained_curve: UtilityCurve,
    spending_curve: UtilityCurve,
    *,
    spending_importance: float = 1.0,
    maximum_spending: float | None = None,
    candidates: int = 65,
    refinements: int = 3,
) -> FloatArray:
    """Allocate a finite resource between retained wealth and one-time spending.

    The optimizer is vectorized and makes no concavity assumption, which matters
    for bounded soft-floor curves. Each refinement searches the neighborhood of
    the prior best candidate.
    """

    available = np.asarray(resources, dtype=float)
    if not np.all(np.isfinite(available)) or np.any(available < 0.0):
        raise ValueError("resources must be finite and nonnegative")
    if not np.isfinite(spending_importance) or spending_importance < 0.0:
        raise ValueError("spending_importance must be finite and nonnegative")
    if maximum_spending is not None and (
        not np.isfinite(maximum_spending) or maximum_spending < 0.0
    ):
        raise ValueError("maximum_spending must be finite and nonnegative")
    if not isinstance(candidates, int) or candidates < 3:
        raise ValueError("candidates must be an integer of at least three")
    if not isinstance(refinements, int) or refinements < 1:
        raise ValueError("refinements must be a positive integer")

    upper = np.array(available, copy=True)
    if maximum_spending is not None:
        np.minimum(upper, maximum_spending, out=upper)
    capacity = np.array(upper, copy=True)
    lower = np.zeros_like(upper)
    fractions = np.linspace(0.0, 1.0, candidates)
    best = np.zeros_like(upper)

    for _ in range(refinements):
        span = upper - lower
        choices = lower[..., np.newaxis] + span[..., np.newaxis] * fractions
        retained = available[..., np.newaxis] - choices
        scores = retained_curve.evaluate(retained)
        scores += spending_importance * spending_curve.evaluate(choices)
        optimum = np.argmax(scores, axis=-1)
        best = np.take_along_axis(
            choices,
            optimum[..., np.newaxis],
            axis=-1,
        )[..., 0]
        step = span / (candidates - 1)
        lower = np.maximum(best - step, 0.0)
        upper = np.minimum(best + step, capacity)

    return np.asarray(best, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class RollingDecision:
    spending: FloatArray
    equity_fraction: FloatArray
    leverage: FloatArray
    insured_bequest: FloatArray
    insurance_premium: FloatArray


@dataclass(frozen=True, slots=True)
class RollingDecisionContext:
    spending: SpendingContext
    existing_insured_bequest: FloatArray
    insurance_price: float
    maximum_leverage: float
    market: MarketModelConfig
    base_allocation: Allocation
    fixed_allocation: bool = False
    leverage_cost: float = 0.0
    effective_leverage: FloatArray | None = None
    leverage_locked: BoolArray | None = None
    active_paths: BoolArray | None = None
    fixed_spending: FloatArray | None = None


class RollingDecisionPolicy(Protocol):
    def decide(self, context: RollingDecisionContext) -> RollingDecision:
        """Jointly choose current controls for each path."""


@dataclass(frozen=True, slots=True)
class JointRollingDecisionOptimizer:
    """Coordinate optimizer for spending, insurance, allocation, and leverage.

    Allocation and leverage maximize a risk-adjusted continuation score plus
    their explicit utility add-ons. Conditional on that exposure, spending
    and additional insured bequest are optimized jointly.
    """

    utility_model: UtilityModel
    equity_candidates: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    leverage_candidates: tuple[float, ...] = (1.0, 1.25, 1.5, 2.0)
    bequest_multipliers: tuple[float, ...] = (1.0,)
    solver: SpendingSolver = field(default_factory=DerivativeSpendingSolver)

    def __post_init__(self) -> None:
        if not self.equity_candidates or not self.leverage_candidates:
            raise ValueError("equity and leverage candidates cannot be empty")
        if any(not 0 <= value <= 1 for value in self.equity_candidates):
            raise ValueError("equity candidates must be between zero and one")
        if any(value < 1 for value in self.leverage_candidates):
            raise ValueError("leverage candidates must be at least one")
        if any(value < 0 for value in self.bequest_multipliers):
            raise ValueError("bequest multipliers cannot be negative")

    def _projected_addon_score(
        self,
        addon: UtilityAddon,
        value: float | FloatArray,
        context: RollingDecisionContext,
        window: int,
    ) -> float:
        offsets = np.arange(window)
        ages = np.arange(
            context.spending.age,
            context.spending.age + window,
        )
        survival = np.asarray(
            context.spending.lifecycle_plan.survival_probabilities[
                context.spending.year : context.spending.year + window
            ],
            dtype=float,
        )
        conditional = survival / max(survival[0], np.finfo(float).tiny)
        weights = (
            (1.0 + self.utility_model.preferences.time_preference) ** -offsets
            * conditional
            * vitality(ages, self.utility_model.preferences)
            * addon.age_profile(ages)
        )
        if addon.minimum_age is not None:
            weights = np.where(ages >= addon.minimum_age, weights, 0.0)
        if addon.maximum_age is not None:
            weights = np.where(ages <= addon.maximum_age, weights, 0.0)
        if weights.sum() <= 0:
            return 0.0
        component = float(np.mean(addon.curve.evaluate(value)))
        if addon.aggregation is UtilityAggregation.DISCOUNTED_SUM:
            component *= float(weights.sum())
        return addon.importance * component

    def _choose_exposure(
        self,
        context: RollingDecisionContext,
    ) -> tuple[float, float, float]:
        preferences = self.utility_model.preferences
        if not np.isfinite(context.leverage_cost) or context.leverage_cost < 0.0:
            raise ValueError("leverage_cost must be finite and nonnegative")
        remaining = context.spending.scenario.person.horizon - context.spending.year
        continuation_window = min(remaining, 10)
        future_income = context.spending.future_income[context.spending.year :]
        base_level = (
            context.spending.wealth
            + context.spending.income
            + max(float(np.sum(future_income[1:])), 0.0)
        ) / max(remaining, 1)
        base_level = np.maximum(base_level, 1.0)
        candidates: list[tuple[float, float, float]] = []
        equity_candidates = (
            (context.base_allocation.equity,)
            if context.fixed_allocation
            else self.equity_candidates
        )
        leverage_candidates = tuple(
            dict.fromkeys(
                (
                    *(
                        value
                        for value in self.leverage_candidates
                        if value <= context.maximum_leverage + 1e-12
                    ),
                    context.maximum_leverage,
                )
            )
        )
        for equity in equity_candidates:
            for leverage in leverage_candidates:
                exposure = equity + leverage - 1.0
                overlay = leverage - 1.0
                certainty_equivalent_spread = (
                    exposure * context.market.equity_risk_premium
                    - overlay * context.leverage_cost
                    - 0.5
                    * (exposure * context.market.equity_volatility) ** 2
                    / preferences.effective_risk_tolerance
                )
                projected_spending = base_level * np.exp(
                    continuation_window * certainty_equivalent_spread
                )
                utility = 0.0
                for addon in self.utility_model.addons:
                    if addon.outcome is OutcomeType.SPENDING:
                        value: float | FloatArray = projected_spending
                    elif addon.outcome is OutcomeType.ALLOCATION_EQUITY:
                        value = equity
                    elif addon.outcome is OutcomeType.LEVERAGE:
                        value = leverage
                    else:
                        continue
                    utility += self._projected_addon_score(
                        addon,
                        value,
                        context,
                        continuation_window,
                    )
                candidates.append((utility, equity, leverage))
        if not candidates:
            equity = context.base_allocation.equity
            return equity, 1.0, 0.0
        _, equity, leverage = max(candidates, key=lambda item: item[0])
        exposure = equity + leverage - 1.0
        overlay = leverage - 1.0
        spread = (
            exposure * context.market.equity_risk_premium
            - overlay * context.leverage_cost
            - 0.5
            * (exposure * context.market.equity_volatility) ** 2
            / preferences.effective_risk_tolerance
        )
        return equity, leverage, spread

    def _decide_all(self, context: RollingDecisionContext) -> RollingDecision:
        spending_context = context.spending
        paths = len(spending_context.wealth)
        equity, leverage, chosen_spread = self._choose_exposure(context)
        expected_spread: float | FloatArray = chosen_spread
        if (context.effective_leverage is None) != (context.leverage_locked is None):
            raise ValueError(
                "effective_leverage and leverage_locked must be supplied together"
            )
        if context.effective_leverage is not None:
            effective_leverage = np.asarray(context.effective_leverage, dtype=float)
            leverage_locked = np.asarray(context.leverage_locked, dtype=bool)
            if (
                effective_leverage.shape != (paths,)
                or leverage_locked.shape != (paths,)
            ):
                raise ValueError(
                    "effective_leverage and leverage_locked must align with spending paths"
                )
            if not np.all(np.isfinite(effective_leverage)) or np.any(
                effective_leverage < 0.0
            ):
                raise ValueError("effective_leverage must be finite and nonnegative")
            accounting_leverage = np.maximum(effective_leverage, 1.0)
            realized_leverage = np.where(
                leverage_locked,
                accounting_leverage,
                leverage,
            )
            realized_exposure = equity + realized_leverage - 1.0
            expected_spread = (
                realized_exposure * context.market.equity_risk_premium
                - (realized_leverage - 1.0) * context.leverage_cost
                - 0.5
                * (realized_exposure * context.market.equity_volatility) ** 2
                / self.utility_model.preferences.effective_risk_tolerance
            )
        horizon = len(spending_context.future_income)
        remaining = horizon - spending_context.year
        ages = np.arange(spending_context.age, spending_context.age + remaining)
        financial_rate = np.maximum(
            spending_context.real_rate + expected_spread,
            -0.99,
        )
        survival = np.asarray(
            spending_context.lifecycle_plan.survival_probabilities[spending_context.year :],
            dtype=float,
        )
        conditional = survival / max(survival[0], np.finfo(float).tiny)
        annuity, income_pv = _discounted_annuity_and_income(
            financial_rate,
            conditional,
            spending_context.future_income[spending_context.year :],
            spending_context.income,
        )
        resources = np.maximum(
            spending_context.wealth + income_pv,
            0.0,
        )

        existing = context.existing_insured_bequest
        available_cash = np.maximum(
            spending_context.wealth + spending_context.income,
            0.0,
        )
        has_bequest_utility = any(
            addon.outcome
            in {
                OutcomeType.BEQUEST,
                OutcomeType.INSURED_BEQUEST,
            }
            for addon in self.utility_model.addons
        )
        if has_bequest_utility:
            spending_reserve = (
                np.minimum(context.fixed_spending, available_cash)
                if context.fixed_spending is not None
                else np.zeros(paths)
            )
            maximum_increase = (
                np.maximum(available_cash - spending_reserve, 0.0)
                / context.insurance_price
                if context.insurance_price > 0
                else np.zeros(paths)
            )
            maximum_bequest = existing + maximum_increase
            plan_bequest = spending_context.lifecycle_plan.bequest
            desired_candidates = [
                existing,
                maximum_bequest,
                *[
                    np.full(paths, plan_bequest * multiplier)
                    for multiplier in self.bequest_multipliers
                ],
            ]
            clipped_candidates: list[FloatArray] = []
            for candidate in desired_candidates:
                clipped = np.clip(candidate, existing, maximum_bequest)
                if not any(
                    np.array_equal(clipped, prior) for prior in clipped_candidates
                ):
                    clipped_candidates.append(clipped)
            desired = np.stack(clipped_candidates, axis=1)
        else:
            desired = existing[:, np.newaxis]
        premium = (desired - existing[:, np.newaxis]) * context.insurance_price
        net_resources = np.maximum(resources[:, np.newaxis] - premium, 0.0)
        candidate_count = desired.shape[1]
        flat_problem = build_spending_problem(
            utility_model=self.utility_model,
            resources=net_resources.reshape(-1),
            future_annuity=np.repeat(annuity, candidate_count),
            ages=ages.astype(float),
            conditional_survival=conditional,
        )
        cash_after_premium = np.maximum(
            available_cash[:, np.newaxis] - premium,
            0.0,
        )
        maximum_spending = np.minimum(
            cash_after_premium.reshape(-1),
            flat_problem.resources,
        )
        if context.fixed_spending is not None:
            current_spending = np.minimum(
                np.repeat(context.fixed_spending, candidate_count),
                maximum_spending,
            )
            spending_score = self.solver.objective(
                flat_problem,
                current_spending[:, np.newaxis],
            )[:, 0]
        else:
            solve_with_score = getattr(self.solver, "solve_with_score", None)
            solver_type = type(self.solver)
            legacy_solve_override = (
                isinstance(self.solver, DerivativeSpendingSolver)
                and solver_type.solve is not DerivativeSpendingSolver.solve
                and getattr(solver_type, "solve_with_score", None)
                is DerivativeSpendingSolver.solve_with_score
            )
            if callable(solve_with_score) and not legacy_solve_override:
                solution = solve_with_score(
                    flat_problem,
                    maximum_spending=maximum_spending,
                )
                current_spending = solution.spending
                spending_score = solution.score
            else:
                current_spending = np.minimum(
                    self.solver.solve(flat_problem),
                    maximum_spending,
                )
                spending_score = self.solver.objective(
                    flat_problem,
                    current_spending[:, np.newaxis],
                )[:, 0]
        bequest_score = np.zeros_like(spending_score)
        flat_bequest = desired.reshape(-1)
        for addon in self.utility_model.addons:
            if addon.outcome in {
                OutcomeType.BEQUEST,
                OutcomeType.INSURED_BEQUEST,
            }:
                bequest_score += addon.importance * addon.curve.evaluate(flat_bequest)
        total = (spending_score + bequest_score).reshape(paths, candidate_count)
        optimum = np.argmax(total, axis=1)
        row = np.arange(paths)
        return RollingDecision(
            spending=current_spending.reshape(paths, candidate_count)[row, optimum],
            equity_fraction=np.full(paths, equity),
            leverage=np.full(paths, leverage),
            insured_bequest=desired[row, optimum],
            insurance_premium=premium[row, optimum],
        )

    def decide(self, context: RollingDecisionContext) -> RollingDecision:
        paths = len(context.spending.wealth)
        if context.fixed_spending is not None:
            fixed_spending = np.asarray(context.fixed_spending, dtype=float)
            if fixed_spending.shape != (paths,):
                raise ValueError("fixed_spending must align with spending paths")
            if not np.all(np.isfinite(fixed_spending)):
                raise ValueError("fixed_spending must be finite")
            if np.any(fixed_spending < 0.0):
                raise ValueError("fixed_spending cannot be negative")
            context = replace(context, fixed_spending=fixed_spending)
        if context.active_paths is None:
            return self._decide_all(context)
        active = np.asarray(context.active_paths, dtype=bool)
        if active.shape != (paths,):
            raise ValueError("active_paths must align with spending paths")
        if np.all(active):
            return self._decide_all(replace(context, active_paths=None))

        active_count = int(np.count_nonzero(active))
        if active_count == 0:
            return RollingDecision(
                spending=np.zeros(paths),
                equity_fraction=np.zeros(paths),
                leverage=np.ones(paths),
                insured_bequest=context.existing_insured_bequest.copy(),
                insurance_premium=np.zeros(paths),
            )

        spending = context.spending
        compact_context = replace(
            context,
            spending=replace(
                spending,
                wealth=spending.wealth[active],
                income=spending.income[active],
                real_rate=spending.real_rate[active],
            ),
            existing_insured_bequest=context.existing_insured_bequest[active],
            effective_leverage=(
                None
                if context.effective_leverage is None
                else context.effective_leverage[active]
            ),
            leverage_locked=(
                None
                if context.leverage_locked is None
                else context.leverage_locked[active]
            ),
            fixed_spending=(
                None
                if context.fixed_spending is None
                else context.fixed_spending[active]
            ),
            active_paths=None,
        )
        compact = self._decide_all(compact_context)
        result = RollingDecision(
            spending=np.zeros(paths),
            equity_fraction=np.zeros(paths),
            leverage=np.ones(paths),
            insured_bequest=context.existing_insured_bequest.copy(),
            insurance_premium=np.zeros(paths),
        )
        result.spending[active] = compact.spending
        result.equity_fraction[active] = compact.equity_fraction
        result.leverage[active] = compact.leverage
        result.insured_bequest[active] = compact.insured_bequest
        result.insurance_premium[active] = compact.insurance_premium
        return result
