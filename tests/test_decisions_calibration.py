from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import lifecycle_finance.decisions as decisions_module
from lifecycle_finance import (
    DerivativeSpendingSolver,
    ImportanceCalibration,
    IsoelasticCurve,
    JointRollingDecisionOptimizer,
    LinearCurve,
    MonteCarloEngine,
    OutcomeType,
    PlanningScenario,
    RollingDecisionContext,
    SimulationSettings,
    SpendingContext,
    SpendingFloorCurve,
    SpendingOptimizationProblem,
    TargetCurve,
    UtilityAddon,
    UtilityAggregation,
    UtilityCalibrator,
    UtilityModel,
    UtilityOutcome,
    optimize_resource_split,
)
from lifecycle_finance.spending import build_spending_problem


def _outcome(
    scenario: PlanningScenario,
    spending: float | np.ndarray,
    *,
    retirement_age: float = 66,
) -> UtilityOutcome:
    values = np.broadcast_to(
        np.asarray(spending, dtype=float),
        (1, scenario.person.horizon),
    ).copy()
    return UtilityOutcome(
        spending=values,
        exposure=np.ones_like(values),
        ages=tuple(
            range(
                scenario.person.current_age,
                scenario.person.maximum_age + 1,
            )
        ),
        terminal_wealth=np.array([1_000_000.0]),
        decisions={OutcomeType.RETIREMENT_AGE: retirement_age},
    )


def test_typed_outcomes_validate_and_aggregation_is_explicit() -> None:
    scenario = replace(
        PlanningScenario(),
        preferences=replace(
            PlanningScenario().preferences,
            bequest_strength=0,
            spending_floor_importance=0,
        ),
    )
    leverage = UtilityAddon(
        "leverage_preference",
        OutcomeType.LEVERAGE,
        TargetCurve(1.25, 0.25),
    )
    model = UtilityModel.from_scenario(scenario, [leverage])
    with pytest.raises(ValueError, match="unavailable"):
        model.validate_outcomes(frozenset({OutcomeType.SPENDING}))
    active = model.validate_outcomes(
        frozenset({OutcomeType.SPENDING}),
        require_all=False,
    )
    assert [addon.name for addon in active] == ["consumption"]

    outcome = UtilityOutcome(
        spending=np.array([[50_000.0, 30_000.0]]),
        exposure=np.ones((1, 2)),
        ages=(65, 66),
        terminal_wealth=np.array([0.0]),
        decisions={},
    )
    curve = SpendingFloorCurve(40_000, 10_000)
    weights = np.ones((1, 2))
    mean = UtilityAddon(
        "mean",
        OutcomeType.SPENDING,
        curve,
        aggregation=UtilityAggregation.DISCOUNTED_MEAN,
    )
    worst = replace(mean, name="worst", aggregation=UtilityAggregation.WORST)
    total = replace(
        mean,
        name="sum",
        aggregation=UtilityAggregation.DISCOUNTED_SUM,
    )
    assert mean.score(outcome, weights)[0] == pytest.approx(-0.5)
    assert worst.score(outcome, weights)[0] == pytest.approx(-1.0)
    assert total.score(outcome, weights)[0] == pytest.approx(-1.0)


def test_wedding_spend_is_a_typed_satiating_utility_outcome() -> None:
    scenario = replace(
        PlanningScenario(),
        preferences=replace(
            PlanningScenario().preferences,
            bequest_strength=0,
            spending_floor_importance=0,
        ),
    )
    wedding = UtilityAddon(
        "wedding",
        OutcomeType.WEDDING_SPEND,
        SpendingFloorCurve(350_000, 100_000),
    )
    model = UtilityModel.from_scenario(scenario, [wedding])
    outcome = _outcome(scenario, 70_000)
    outcome = replace(
        outcome,
        decisions={OutcomeType.WEDDING_SPEND: 250_000},
    )

    model.validate_outcomes(outcome.available_outcomes)
    wedding_score = wedding.score(outcome, model.weights(outcome))
    assert wedding_score[0] == pytest.approx(-1.0)


def test_resource_split_optimizer_matches_dense_bounded_utility_search() -> None:
    resources = np.array([200_000.0, 1_000_000.0, 2_250_000.0])
    retained = IsoelasticCurve(2_000_000.0, 0.4, minimum_utility=-10.0)
    wedding = SpendingFloorCurve(
        350_000.0,
        100_000.0,
        curvature=2.949908233285761,
    )
    importance = 0.05303536709176626
    result = optimize_resource_split(
        resources,
        retained,
        wedding,
        spending_importance=importance,
        maximum_spending=350_000.0,
    )

    fractions = np.linspace(0.0, 1.0, 100_001)
    dense_choices = np.minimum(resources[:, np.newaxis], 350_000.0) * fractions
    scores = retained.evaluate(resources[:, np.newaxis] - dense_choices)
    scores += importance * wedding.evaluate(dense_choices)
    dense = dense_choices[np.arange(len(resources)), np.argmax(scores, axis=1)]
    np.testing.assert_allclose(result, dense, atol=5.0)


def test_resource_split_optimizer_validates_inputs() -> None:
    curve = LinearCurve()
    with pytest.raises(ValueError, match="resources"):
        optimize_resource_split([-1.0], curve, curve)
    with pytest.raises(ValueError, match="spending_importance"):
        optimize_resource_split([1.0], curve, curve, spending_importance=-1.0)
    with pytest.raises(ValueError, match="candidates"):
        optimize_resource_split([1.0], curve, curve, candidates=2)


def test_derivative_solver_matches_dense_objective_search() -> None:
    addons = (
        UtilityAddon(
            "consumption",
            OutcomeType.SPENDING,
            IsoelasticCurve(40_000, 0.4),
        ),
        UtilityAddon(
            "floor",
            OutcomeType.SPENDING,
            SpendingFloorCurve(40_000, 10_000),
            importance=2,
        ),
    )
    problem = SpendingOptimizationProblem(
        resources=np.array([1_000_000.0, 2_000_000.0]),
        future_annuity=np.array([15.0, 20.0]),
        addons=addons,
        current_weights=np.array([1.0, 1.0]),
        future_weights=np.array([10.0, 10.0]),
        aggregations=(
            UtilityAggregation.DISCOUNTED_MEAN,
            UtilityAggregation.DISCOUNTED_MEAN,
        ),
    )
    solver = DerivativeSpendingSolver(iterations=20)
    result = solver.solve(problem)
    fractions = np.linspace(0, 1, 20_001)
    dense_candidates = problem.resources[:, np.newaxis] * fractions
    dense_scores = solver.objective(problem, dense_candidates)
    dense = dense_candidates[
        np.arange(len(problem.resources)),
        np.argmax(dense_scores, axis=1),
    ]
    np.testing.assert_allclose(result, dense, rtol=5e-4, atol=10.0)


def test_derivative_solver_returns_selected_objective_score() -> None:
    problem = SpendingOptimizationProblem(
        resources=np.array([1_000_000.0, 2_000_000.0]),
        future_annuity=np.array([15.0, 20.0]),
        addons=(
            UtilityAddon(
                "consumption",
                OutcomeType.SPENDING,
                IsoelasticCurve(40_000, 0.4),
            ),
            UtilityAddon(
                "floor",
                OutcomeType.SPENDING,
                SpendingFloorCurve(40_000, 10_000),
                importance=2,
            ),
        ),
        current_weights=np.array([1.0, 1.0]),
        future_weights=np.array([10.0, 10.0]),
        aggregations=(
            UtilityAggregation.DISCOUNTED_MEAN,
            UtilityAggregation.DISCOUNTED_MEAN,
        ),
    )
    solver = DerivativeSpendingSolver()

    solution = solver.solve_with_score(problem)
    direct_score = solver.objective(problem, solution.spending[:, np.newaxis])[:, 0]

    np.testing.assert_array_equal(solution.score, direct_score)
    np.testing.assert_array_equal(solver.solve(problem), solution.spending)


def test_derivative_solver_builtin_curves_match_fixed_bisection() -> None:
    class GenericIsoelastic(IsoelasticCurve):
        def marginal_utility(self, value: object) -> np.ndarray:
            return IsoelasticCurve.marginal_utility(self, value)

        def marginal_utility_into(self, value: object, out: np.ndarray) -> None:
            IsoelasticCurve.marginal_utility_into(self, value, out)

    class GenericTarget(TargetCurve):
        def marginal_utility(self, value: object) -> np.ndarray:
            return TargetCurve.marginal_utility(self, value)

        def marginal_utility_into(self, value: object, out: np.ndarray) -> None:
            TargetCurve.marginal_utility_into(self, value, out)

    target = UtilityAddon(
        "target",
        OutcomeType.SPENDING,
        TargetCurve(
            target=1.037182612e-5,
            tolerance=172.525405,
            curvature=10.0,
        ),
    )
    isoelastic = UtilityAddon(
        "isoelastic",
        OutcomeType.SPENDING,
        IsoelasticCurve(1.0, 0.75, minimum_utility=-10.0),
        importance=1e-300,
    )
    problem = SpendingOptimizationProblem(
        resources=np.array([0.0117892071, 8_563_915.8, 427.367018]),
        future_annuity=np.array([0.0195347448, 33_194.3833, 49_892.9756]),
        addons=(target, isoelastic),
        current_weights=np.array([0.0255997442, 1.0]),
        future_weights=np.array([0.000122969781, 1.0]),
        aggregations=(UtilityAggregation.DISCOUNTED_SUM,) * 2,
    )
    generic_problem = replace(
        problem,
        addons=(
            replace(
                target,
                curve=GenericTarget(
                    target=1.037182612e-5,
                    tolerance=172.525405,
                    curvature=10.0,
                ),
            ),
            replace(
                isoelastic,
                curve=GenericIsoelastic(
                    1.0,
                    0.75,
                    minimum_utility=-10.0,
                ),
            ),
        ),
    )
    maximum = np.array([0.00735445998, 3_803_260.23, 89.3412194])
    solver = DerivativeSpendingSolver()

    builtin = solver.solve_with_score(problem, maximum_spending=maximum)
    generic = solver.solve_with_score(generic_problem, maximum_spending=maximum)

    np.testing.assert_array_equal(builtin.spending, generic.spending)
    np.testing.assert_array_equal(builtin.score, generic.score)


def test_derivative_solver_enforces_per_path_maximum_spending() -> None:
    problem = SpendingOptimizationProblem(
        resources=np.array([100_000.0, 200_000.0]),
        future_annuity=np.ones(2),
        addons=(
            UtilityAddon(
                "consumption",
                OutcomeType.SPENDING,
                IsoelasticCurve(40_000, 1.0),
            ),
        ),
        current_weights=np.ones(1),
        future_weights=np.ones(1),
        aggregations=(UtilityAggregation.DISCOUNTED_MEAN,),
    )
    maximum_spending = np.array([25_000.0, 75_000.0])

    solution = DerivativeSpendingSolver().solve_with_score(
        problem,
        maximum_spending=maximum_spending,
    )

    np.testing.assert_array_equal(solution.spending, maximum_spending)


def test_derivative_solver_deduplicates_equal_numeric_breakpoints() -> None:
    objective_column_counts: list[int] = []

    class RecordingSolver(DerivativeSpendingSolver):
        @staticmethod
        def objective(
            problem: SpendingOptimizationProblem,
            candidates: np.ndarray,
        ) -> np.ndarray:
            objective_column_counts.append(candidates.shape[1])
            return DerivativeSpendingSolver.objective(problem, candidates)

    problem = SpendingOptimizationProblem(
        resources=np.array([100_000.0]),
        future_annuity=np.array([1.0]),
        addons=(
            UtilityAddon(
                "consumption",
                OutcomeType.SPENDING,
                IsoelasticCurve(40_000, 0.4),
            ),
            UtilityAddon(
                "floor",
                OutcomeType.SPENDING,
                SpendingFloorCurve(40_000, 10_000),
            ),
        ),
        current_weights=np.ones(2),
        future_weights=np.ones(2),
        aggregations=(UtilityAggregation.DISCOUNTED_MEAN,) * 2,
    )

    RecordingSolver().solve_with_score(problem)

    assert objective_column_counts == [5]


def test_derivative_solver_preserves_first_seen_breakpoint_order() -> None:
    class TwoPeakCurve:
        def evaluate(self, value: object) -> np.ndarray:
            values = np.asarray(value, dtype=float)
            return -((values - 20.0) * (values - 80.0)) ** 2

        def marginal_utility(self, value: object) -> np.ndarray:
            values = np.asarray(value, dtype=float)
            return -2.0 * (values - 20.0) * (values - 80.0) * (2.0 * values - 100.0)

        def breakpoints(self) -> tuple[float, ...]:
            return (80.0, 20.0)

    problem = SpendingOptimizationProblem(
        resources=np.array([100.0]),
        future_annuity=np.array([0.0]),
        addons=(UtilityAddon("two_peaks", OutcomeType.SPENDING, TwoPeakCurve()),),
        current_weights=np.array([1.0]),
        future_weights=np.array([0.0]),
        aggregations=(UtilityAggregation.DISCOUNTED_MEAN,),
    )

    result = DerivativeSpendingSolver().solve(problem)

    np.testing.assert_array_equal(result, [80.0])


def test_derivative_solver_handles_linear_spending_floor_flat_branch() -> None:
    problem = SpendingOptimizationProblem(
        resources=np.array([100_000.0]),
        future_annuity=np.array([1.0]),
        addons=(
            UtilityAddon(
                "consumption",
                OutcomeType.SPENDING,
                IsoelasticCurve(40_000, 0.4),
            ),
            UtilityAddon(
                "floor",
                OutcomeType.SPENDING,
                SpendingFloorCurve(30_000, 10_000, curvature=1.0),
                importance=0.5,
            ),
        ),
        current_weights=np.array([0.5, 0.5]),
        future_weights=np.array([1.0, 1.0]),
        aggregations=(
            UtilityAggregation.DISCOUNTED_MEAN,
            UtilityAggregation.DISCOUNTED_MEAN,
        ),
    )

    result = DerivativeSpendingSolver(iterations=24).solve(problem)

    assert result[0] == pytest.approx(43_115.0, abs=25.0)


def test_derivative_solver_uses_builtin_marginal_fast_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CurveProxy:
        def __init__(self, curve: object, marginal: object) -> None:
            self.curve = curve
            self.marginal = marginal

        def evaluate(self, value: object) -> np.ndarray:
            return self.curve.evaluate(value)  # type: ignore[attr-defined, no-any-return]

        def marginal_utility(self, value: object) -> np.ndarray:
            return self.marginal(self.curve, value)  # type: ignore[operator, no-any-return]

        def breakpoints(self) -> tuple[float, ...]:
            return self.curve.breakpoints()  # type: ignore[attr-defined, no-any-return]

    isoelastic = IsoelasticCurve(40_000, 0.4)
    floor = SpendingFloorCurve(40_000, 10_000)
    original_isoelastic = IsoelasticCurve.marginal_utility
    original_floor = SpendingFloorCurve.marginal_utility
    calls = {"isoelastic": 0, "floor": 0}

    def counted_isoelastic(self: IsoelasticCurve, value: object) -> np.ndarray:
        calls["isoelastic"] += 1
        return original_isoelastic(self, value)

    def counted_floor(self: SpendingFloorCurve, value: object) -> np.ndarray:
        calls["floor"] += 1
        return original_floor(self, value)

    monkeypatch.setattr(IsoelasticCurve, "marginal_utility", counted_isoelastic)
    monkeypatch.setattr(SpendingFloorCurve, "marginal_utility", counted_floor)
    problem = SpendingOptimizationProblem(
        resources=np.array([1_000_000.0, 2_000_000.0]),
        future_annuity=np.array([15.0, 20.0]),
        addons=(
            UtilityAddon("consumption", OutcomeType.SPENDING, isoelastic),
            UtilityAddon("floor", OutcomeType.SPENDING, floor, importance=2),
        ),
        current_weights=np.array([1.0, 1.0]),
        future_weights=np.array([10.0, 10.0]),
        aggregations=(
            UtilityAggregation.DISCOUNTED_MEAN,
            UtilityAggregation.DISCOUNTED_MEAN,
        ),
    )

    solver = DerivativeSpendingSolver()
    result = solver.solve(problem)
    generic_problem = replace(
        problem,
        addons=(
            replace(
                problem.addons[0],
                curve=CurveProxy(isoelastic, original_isoelastic),
            ),
            replace(
                problem.addons[1],
                curve=CurveProxy(floor, original_floor),
            ),
        ),
    )
    generic_result = solver.solve(generic_problem)

    assert calls == {"isoelastic": 0, "floor": 0}
    np.testing.assert_array_equal(result, generic_result)


@pytest.mark.parametrize("aggregation", list(UtilityAggregation))
def test_objective_uses_builtin_curve_fast_path(
    aggregation: UtilityAggregation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CurveProxy:
        def __init__(self, curve: object, evaluate: object) -> None:
            self.curve = curve
            self.evaluate_function = evaluate

        def evaluate(self, value: object) -> np.ndarray:
            return self.evaluate_function(  # type: ignore[operator, no-any-return]
                self.curve,
                value,
            )

    curves = (
        IsoelasticCurve(40_000, 0.4),
        SpendingFloorCurve(40_000, 10_000),
        TargetCurve(70_000, 20_000),
        LinearCurve(0.001),
    )
    curve_types = (IsoelasticCurve, SpendingFloorCurve, TargetCurve, LinearCurve)
    originals = tuple(curve_type.evaluate for curve_type in curve_types)
    calls = dict.fromkeys(curve_types, 0)

    for curve_type, original in zip(curve_types, originals, strict=True):

        def counted(
            self: object,
            value: object,
            *,
            _curve_type: type[object] = curve_type,
            _original: object = original,
        ) -> np.ndarray:
            calls[_curve_type] += 1
            return _original(self, value)  # type: ignore[operator, no-any-return]

        monkeypatch.setattr(curve_type, "evaluate", counted)

    addons = tuple(
        UtilityAddon(
            f"curve_{index}",
            OutcomeType.SPENDING,
            curve,
            importance=float(index + 1),
            aggregation=aggregation,
        )
        for index, curve in enumerate(curves)
    )
    problem = SpendingOptimizationProblem(
        resources=np.array([1_000_000.0, 2_000_000.0]),
        future_annuity=np.array([15.0, 20.0]),
        addons=addons,
        current_weights=np.ones(len(addons)),
        future_weights=np.full(len(addons), 10.0),
        aggregations=(aggregation,) * len(addons),
    )
    candidates = np.array(
        [
            [30_000.0, 50_000.0, 80_000.0],
            [40_000.0, 75_000.0, 120_000.0],
        ]
    )
    solver = DerivativeSpendingSolver()

    result = solver.objective(problem, candidates)
    generic_problem = replace(
        problem,
        addons=tuple(
            replace(addon, curve=CurveProxy(curve, original))
            for addon, curve, original in zip(
                addons,
                curves,
                originals,
                strict=True,
            )
        ),
    )
    generic_result = solver.objective(generic_problem, candidates)

    assert calls == dict.fromkeys(curve_types, 0)
    np.testing.assert_array_equal(result, generic_result)


def test_builtin_subclass_evaluate_override_is_not_bypassed() -> None:
    class CustomLinear(LinearCurve):
        def evaluate(self, value: object) -> np.ndarray:
            values = np.asarray(value, dtype=float)
            return -((values - 25.0) ** 2)

    problem = SpendingOptimizationProblem(
        resources=np.array([100.0]),
        future_annuity=np.array([1.0]),
        addons=(UtilityAddon("custom", OutcomeType.SPENDING, CustomLinear()),),
        current_weights=np.array([1.0]),
        future_weights=np.array([0.0]),
        aggregations=(UtilityAggregation.DISCOUNTED_MEAN,),
    )

    scores = DerivativeSpendingSolver.objective(
        problem,
        np.array([[0.0, 25.0, 100.0]]),
    )

    np.testing.assert_array_equal(scores, [[-625.0, 0.0, -5_625.0]])


def test_builtin_subclass_marginal_override_is_not_bypassed() -> None:
    class CustomLinear(LinearCurve):
        def evaluate(self, value: object) -> np.ndarray:
            values = np.asarray(value, dtype=float)
            return -((values - 50.0) ** 2)

        def marginal_utility(self, value: object) -> np.ndarray:
            return 100.0 - 2.0 * np.asarray(value, dtype=float)

    problem = SpendingOptimizationProblem(
        resources=np.array([100.0]),
        future_annuity=np.array([1.0]),
        addons=(UtilityAddon("custom", OutcomeType.SPENDING, CustomLinear()),),
        current_weights=np.array([1.0]),
        future_weights=np.array([1.0]),
        aggregations=(UtilityAggregation.DISCOUNTED_MEAN,),
    )

    result = DerivativeSpendingSolver().solve(problem)

    assert result[0] == pytest.approx(50.0, abs=0.01)


def test_custom_curve_without_derivative_uses_grid_fallback() -> None:
    class CustomCurve:
        def evaluate(self, value: object) -> np.ndarray:
            values = np.asarray(value, dtype=float)
            return -(((values - 50_000.0) / 10_000.0) ** 2)

    problem = SpendingOptimizationProblem(
        resources=np.array([100_000.0]),
        future_annuity=np.array([0.0]),
        addons=(
            UtilityAddon(
                "custom",
                OutcomeType.SPENDING,
                CustomCurve(),
            ),
        ),
        current_weights=np.array([1.0]),
        future_weights=np.array([0.0]),
        aggregations=(UtilityAggregation.DISCOUNTED_MEAN,),
    )
    result = DerivativeSpendingSolver(
        fallback_grid_size=101,
    ).solve(problem)
    assert result[0] == pytest.approx(50_000.0)


def test_build_spending_problem_selects_age_active_addons() -> None:
    scenario = PlanningScenario()
    preferences = replace(
        scenario.preferences,
        time_preference=0.0,
        vitality_floor=1.0,
    )
    model = UtilityModel(
        person=scenario.person,
        preferences=preferences,
        addons=(
            UtilityAddon(
                "early_spending",
                OutcomeType.SPENDING,
                LinearCurve(1.0),
                maximum_age=46,
            ),
            UtilityAddon(
                "late_spending",
                OutcomeType.SPENDING,
                LinearCurve(1.0),
                minimum_age=47,
            ),
            UtilityAddon(
                "allocation",
                OutcomeType.ALLOCATION_EQUITY,
                LinearCurve(1.0),
            ),
        ),
    )

    problem = build_spending_problem(
        utility_model=model,
        resources=np.array([100_000.0, 200_000.0]),
        future_annuity=np.array([10.0, 12.0]),
        ages=np.array([45.0, 46.0, 47.0, 48.0]),
        conditional_survival=np.array([1.0, 0.8, 0.6, 0.4]),
    )

    assert [addon.name for addon in problem.addons] == [
        "early_spending",
        "late_spending",
    ]
    np.testing.assert_allclose(problem.current_weights, [1.0, 0.0])
    np.testing.assert_allclose(problem.future_weights, [0.8, 1.0])


def test_joint_optimizer_exposes_rolling_decisions() -> None:
    leverage_preference = UtilityAddon(
        "leverage_preference",
        OutcomeType.LEVERAGE,
        TargetCurve(1.25, 0.1),
        importance=100,
    )
    result = MonteCarloEngine(
        utility_addons=[leverage_preference],
    ).simulate(
        settings=SimulationSettings(
            paths=8,
            seed=9,
            leverage=1.5,
            stochastic_lifespan=False,
        )
    )

    leverage = result.decision_paths[OutcomeType.LEVERAGE]
    insured = result.decision_paths[OutcomeType.INSURED_BEQUEST]
    assert leverage.shape == result.spending_paths.shape
    assert np.all(leverage[:, 0] == pytest.approx(1.25))
    assert np.all(leverage <= 1.5)
    assert np.all(insured >= 0)


def test_rolling_optimizer_always_considers_the_configured_leverage_cap() -> None:
    leverage_preference = UtilityAddon(
        "leverage_preference",
        OutcomeType.LEVERAGE,
        TargetCurve(1.6, 0.01),
        importance=1_000.0,
    )

    result = MonteCarloEngine(
        utility_addons=[leverage_preference],
    ).simulate(
        settings=SimulationSettings(
            paths=4,
            seed=17,
            leverage=1.6,
            stochastic_lifespan=False,
            stochastic_income=False,
        ),
    )

    np.testing.assert_allclose(
        result.decision_paths[OutcomeType.LEVERAGE][:, 0],
        1.6,
    )


def test_locked_leverage_uses_the_current_exposure_for_spending_decisions() -> None:
    scenario = PlanningScenario()
    scenario = replace(
        scenario,
        preferences=replace(
            scenario.preferences,
            bequest_strength=0.0,
            spending_floor_importance=0.0,
        ),
    )
    engine = MonteCarloEngine()
    plan = engine.planner.plan(scenario)
    leverage_preference = UtilityAddon(
        "leverage_preference",
        OutcomeType.LEVERAGE,
        TargetCurve(2.0, 0.01),
        importance=1_000.0,
    )
    optimizer = JointRollingDecisionOptimizer(
        UtilityModel.from_scenario(scenario, [leverage_preference]),
    )

    mixed = optimizer.decide(
        RollingDecisionContext(
            spending=SpendingContext(
                year=0,
                wealth=np.full(2, 1_000_000.0),
                income=np.full(2, 50_000.0),
                future_income=np.asarray(plan.income_path),
                real_rate=np.full(2, 0.02),
                scenario=scenario,
                lifecycle_plan=plan,
            ),
            existing_insured_bequest=np.zeros(2),
            insurance_price=0.0,
            maximum_leverage=2.0,
            market=engine.market.config,
            base_allocation=plan.constrained_allocation,
            fixed_allocation=True,
            effective_leverage=np.array([1.0, 0.5]),
            leverage_locked=np.array([False, True]),
        )
    )
    unlevered = optimizer.decide(
        RollingDecisionContext(
            spending=SpendingContext(
                year=0,
                wealth=np.array([1_000_000.0]),
                income=np.array([50_000.0]),
                future_income=np.asarray(plan.income_path),
                real_rate=np.array([0.02]),
                scenario=scenario,
                lifecycle_plan=plan,
            ),
            existing_insured_bequest=np.zeros(1),
            insurance_price=0.0,
            maximum_leverage=1.0,
            market=engine.market.config,
            base_allocation=plan.constrained_allocation,
            fixed_allocation=True,
        )
    )

    assert mixed.leverage[1] == 2.0
    assert mixed.spending[1] == pytest.approx(unlevered.spending[0])


def test_joint_optimizer_skips_inactive_paths() -> None:
    class RecordingSolver:
        resources: list[np.ndarray]

        def __init__(self) -> None:
            self.resources = []

        def solve(self, problem: SpendingOptimizationProblem) -> np.ndarray:
            self.resources.append(problem.resources.copy())
            return np.full_like(problem.resources, 123.0)

        def objective(
            self,
            problem: SpendingOptimizationProblem,
            candidates: np.ndarray,
        ) -> np.ndarray:
            return np.zeros_like(candidates)

    scenario = PlanningScenario()
    engine = MonteCarloEngine()
    plan = engine.planner.plan(scenario)
    solver = RecordingSolver()
    optimizer = JointRollingDecisionOptimizer(
        UtilityModel.from_scenario(scenario),
        solver=solver,
    )
    decision = optimizer.decide(
        RollingDecisionContext(
            spending=SpendingContext(
                year=0,
                wealth=np.array([1_000_000.0, 1e12, 2_000_000.0]),
                income=np.array([50_000.0, 50_000.0, 50_000.0]),
                future_income=np.asarray(plan.income_path),
                real_rate=np.full(3, 0.02),
                scenario=scenario,
                lifecycle_plan=plan,
            ),
            existing_insured_bequest=np.zeros(3),
            insurance_price=0.0,
            maximum_leverage=1.0,
            market=engine.market.config,
            base_allocation=plan.constrained_allocation,
            active_paths=np.array([True, False, True]),
        )
    )

    assert all(np.max(resources) < 1e12 for resources in solver.resources)
    np.testing.assert_array_equal(decision.spending, [123.0, 0.0, 123.0])
    np.testing.assert_array_equal(decision.equity_fraction[[1]], [0.0])
    np.testing.assert_array_equal(decision.leverage[[1]], [1.0])
    np.testing.assert_array_equal(decision.insurance_premium[[1]], [0.0])


def test_fixed_spending_bypasses_solver_but_keeps_insurance_and_exposure() -> None:
    class FixedSpendingSolver:
        def solve(self, problem: SpendingOptimizationProblem) -> np.ndarray:
            raise AssertionError("fixed spending must bypass solve")

        @staticmethod
        def objective(
            problem: SpendingOptimizationProblem,
            candidates: np.ndarray,
        ) -> np.ndarray:
            return DerivativeSpendingSolver.objective(problem, candidates)

    scenario = PlanningScenario()
    engine = MonteCarloEngine()
    plan = engine.planner.plan(scenario)
    optimizer = JointRollingDecisionOptimizer(
        UtilityModel.from_scenario(
            scenario,
            [
                UtilityAddon(
                    "insured_bequest_preference",
                    OutcomeType.INSURED_BEQUEST,
                    LinearCurve(),
                    importance=1_000.0,
                ),
                UtilityAddon(
                    "equity_preference",
                    OutcomeType.ALLOCATION_EQUITY,
                    TargetCurve(0.75, 0.01),
                    importance=1_000.0,
                ),
            ],
        ),
        solver=FixedSpendingSolver(),
    )

    decision = optimizer.decide(
        RollingDecisionContext(
            spending=SpendingContext(
                year=0,
                wealth=np.array([100_000.0]),
                income=np.zeros(1),
                future_income=np.asarray(plan.income_path),
                real_rate=np.array([0.02]),
                scenario=scenario,
                lifecycle_plan=plan,
            ),
            existing_insured_bequest=np.zeros(1),
            insurance_price=0.5,
            maximum_leverage=1.0,
            market=engine.market.config,
            base_allocation=plan.constrained_allocation,
            fixed_spending=np.array([10_000.0]),
        )
    )

    np.testing.assert_array_equal(decision.spending, [10_000.0])
    assert decision.insurance_premium[0] > 0.0
    np.testing.assert_array_equal(decision.equity_fraction, [0.75])


def test_joint_optimizer_preserves_legacy_solve_override() -> None:
    solve_calls: list[np.ndarray] = []

    class LegacyOverrideSolver(DerivativeSpendingSolver):
        def solve(self, problem: SpendingOptimizationProblem) -> np.ndarray:
            solve_calls.append(problem.resources.copy())
            return np.full_like(problem.resources, 123.0)

    scenario = PlanningScenario()
    scenario = replace(
        scenario,
        preferences=replace(scenario.preferences, bequest_strength=0.0),
    )
    engine = MonteCarloEngine()
    plan = engine.planner.plan(scenario)
    optimizer = JointRollingDecisionOptimizer(
        UtilityModel.from_scenario(scenario),
        solver=LegacyOverrideSolver(),
    )

    decision = optimizer.decide(
        RollingDecisionContext(
            spending=SpendingContext(
                year=0,
                wealth=np.array([100_000.0]),
                income=np.zeros(1),
                future_income=np.asarray(plan.income_path),
                real_rate=np.array([0.02]),
                scenario=scenario,
                lifecycle_plan=plan,
            ),
            existing_insured_bequest=np.zeros(1),
            insurance_price=0.5,
            maximum_leverage=1.0,
            market=engine.market.config,
            base_allocation=plan.constrained_allocation,
        )
    )

    assert len(solve_calls) == 1
    np.testing.assert_array_equal(decision.spending, [123.0])


def test_joint_optimizer_prunes_insurance_without_bequest_utility() -> None:
    objective_rows: list[int] = []

    class RecordingSolver:
        def solve(self, problem: SpendingOptimizationProblem) -> np.ndarray:
            raise AssertionError("fixed spending must bypass solve")

        @staticmethod
        def objective(
            problem: SpendingOptimizationProblem,
            candidates: np.ndarray,
        ) -> np.ndarray:
            objective_rows.append(len(problem.resources))
            return DerivativeSpendingSolver.objective(problem, candidates)

    scenario = PlanningScenario()
    scenario = replace(
        scenario,
        preferences=replace(scenario.preferences, bequest_strength=0.0),
    )
    engine = MonteCarloEngine()
    plan = engine.planner.plan(scenario)
    optimizer = JointRollingDecisionOptimizer(
        UtilityModel.from_scenario(scenario),
        solver=RecordingSolver(),
    )

    decision = optimizer.decide(
        RollingDecisionContext(
            spending=SpendingContext(
                year=0,
                wealth=np.array([100_000.0, 200_000.0]),
                income=np.zeros(2),
                future_income=np.asarray(plan.income_path),
                real_rate=np.full(2, 0.02),
                scenario=scenario,
                lifecycle_plan=plan,
            ),
            existing_insured_bequest=np.array([5_000.0, 10_000.0]),
            insurance_price=0.5,
            maximum_leverage=1.0,
            market=engine.market.config,
            base_allocation=plan.constrained_allocation,
            fixed_spending=np.full(2, 20_000.0),
        )
    )

    assert objective_rows == [2]
    np.testing.assert_array_equal(decision.insured_bequest, [5_000.0, 10_000.0])
    np.testing.assert_array_equal(decision.insurance_premium, np.zeros(2))


def test_joint_optimizer_slices_fixed_spending_to_active_paths() -> None:
    class FixedSpendingSolver:
        def solve(self, problem: SpendingOptimizationProblem) -> np.ndarray:
            raise AssertionError("fixed spending must bypass solve")

        @staticmethod
        def objective(
            problem: SpendingOptimizationProblem,
            candidates: np.ndarray,
        ) -> np.ndarray:
            return DerivativeSpendingSolver.objective(problem, candidates)

    scenario = PlanningScenario()
    scenario = replace(
        scenario,
        preferences=replace(scenario.preferences, bequest_strength=0.0),
    )
    engine = MonteCarloEngine()
    plan = engine.planner.plan(scenario)
    optimizer = JointRollingDecisionOptimizer(
        UtilityModel.from_scenario(scenario),
        solver=FixedSpendingSolver(),
    )

    decision = optimizer.decide(
        RollingDecisionContext(
            spending=SpendingContext(
                year=0,
                wealth=np.full(3, 100_000.0),
                income=np.zeros(3),
                future_income=np.asarray(plan.income_path),
                real_rate=np.full(3, 0.02),
                scenario=scenario,
                lifecycle_plan=plan,
            ),
            existing_insured_bequest=np.zeros(3),
            insurance_price=0.5,
            maximum_leverage=1.0,
            market=engine.market.config,
            base_allocation=plan.constrained_allocation,
            active_paths=np.array([True, False, True]),
            fixed_spending=np.array([10_000.0, 1e12, 30_000.0]),
        )
    )

    np.testing.assert_array_equal(decision.spending, [10_000.0, 0.0, 30_000.0])


def test_joint_optimizer_uses_discounted_resource_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = PlanningScenario()
    engine = MonteCarloEngine()
    plan = engine.planner.plan(scenario)
    optimizer = JointRollingDecisionOptimizer(UtilityModel.from_scenario(scenario))

    original = decisions_module._discounted_annuity_and_income
    calls: list[tuple[tuple[int, ...], ...]] = []

    def recording_kernel(
        real_rate: np.ndarray,
        conditional_survival: np.ndarray,
        future_income: np.ndarray,
        current_income: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        calls.append(
            (
                real_rate.shape,
                conditional_survival.shape,
                future_income.shape,
                current_income.shape,
            )
        )
        return original(
            real_rate,
            conditional_survival,
            future_income,
            current_income,
        )

    monkeypatch.setattr(
        decisions_module,
        "_discounted_annuity_and_income",
        recording_kernel,
    )
    decision = optimizer.decide(
        RollingDecisionContext(
            spending=SpendingContext(
                year=0,
                wealth=np.array([1_000_000.0, 2_000_000.0]),
                income=np.array([50_000.0, 75_000.0]),
                future_income=np.asarray(plan.income_path),
                real_rate=np.array([0.01, 0.03]),
                scenario=scenario,
                lifecycle_plan=plan,
            ),
            existing_insured_bequest=np.zeros(2),
            insurance_price=0.1,
            maximum_leverage=1.0,
            market=engine.market.config,
            base_allocation=plan.constrained_allocation,
        )
    )

    assert np.all(np.isfinite(decision.spending))
    assert calls == [
        (
            (2,),
            (scenario.person.horizon,),
            (scenario.person.horizon,),
            (2,),
        )
    ]


def test_pairwise_calibration_recovers_indifference_weight() -> None:
    scenario = replace(
        PlanningScenario(),
        preferences=replace(
            PlanningScenario().preferences,
            bequest_strength=0,
            spending_floor_importance=0,
        ),
    )
    retirement = UtilityAddon(
        "retirement_timing",
        OutcomeType.RETIREMENT_AGE,
        TargetCurve(65, 2),
        importance=1,
    )
    model = UtilityModel.from_scenario(scenario, [retirement])
    calibrator = UtilityCalibrator(model)
    higher_spending_later = _outcome(
        scenario,
        60_000,
        retirement_age=69,
    )
    lower_spending_earlier = _outcome(
        scenario,
        50_000,
        retirement_age=65,
    )

    calibration = calibrator.calibrate_importance(
        "retirement_timing",
        higher_spending_later,
        lower_spending_earlier,
    )
    assert isinstance(calibration, ImportanceCalibration)
    assert calibration.importance > 0
    calibrated = UtilityCalibrator(calibrator.with_calibrated_importance(calibration))
    assert np.mean(calibrated.model.score(higher_spending_later)) == pytest.approx(
        np.mean(calibrated.model.score(lower_spending_earlier))
    )

    varying = _outcome(
        scenario,
        np.linspace(40_000, 80_000, scenario.person.horizon),
        retirement_age=65,
    )
    equivalent = calibrated.equivalent_constant_spending(varying)
    assert 40_000 < equivalent < 80_000

    consumption = calibrated.model.addons[0]
    retirement_timing = calibrated.model.addons[-1]
    marginal_trade = calibrated.marginal_rate_of_substitution(
        consumption,
        50_000,
        retirement_timing,
        67,
    )
    assert np.isfinite(marginal_trade)
    assert marginal_trade > 0
