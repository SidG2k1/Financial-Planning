from dataclasses import replace
from typing import Self

import numpy as np
import pytest

import lifecycle_finance.simulation as simulation_module
from lifecycle_finance import (
    Allocation,
    FixedSpending,
    LeverageInstrument,
    LeverageTerms,
    LifecyclePlanner,
    LinearCurve,
    MonteCarloEngine,
    OutcomeType,
    PlanningScenario,
    RegimeSwitchingMarket,
    RollingDecision,
    SimulationSettings,
    TargetCurve,
    UtilityAddon,
    UtilityAggregation,
    leveraged_portfolio_return,
)
from lifecycle_finance.domain import MarketModelConfig
from lifecycle_finance.income_risk import (
    IncomeRiskContext,
    IncomeRiskState,
    IncomeRiskStep,
    PersistentDisplacementIncomeRisk,
)
from lifecycle_finance.return_models import MarketPaths, RegimeModelConfig
from lifecycle_finance.spending import SpendingContext
from lifecycle_finance.sweeps import parameter_sweep, parameter_sweep_2d


class RecordingRegimeMarket(RegimeSwitchingMarket):
    """Regime model that records reconfiguration and concrete use by sweeps."""

    def __init__(
        self,
        config: MarketModelConfig | None = None,
        regime_config: RegimeModelConfig | None = None,
        bond_duration: float = 6.0,
        with_config_calls: list[MarketModelConfig] | None = None,
        generated_types: list[type[RegimeSwitchingMarket]] | None = None,
        ancestry: tuple[MarketModelConfig, ...] = (),
        generated_ancestries: list[tuple[MarketModelConfig, ...]] | None = None,
    ) -> None:
        super().__init__(config, regime_config, bond_duration)
        self.with_config_calls = [] if with_config_calls is None else with_config_calls
        self.generated_types = [] if generated_types is None else generated_types
        self.ancestry = ancestry
        self.generated_ancestries = [] if generated_ancestries is None else generated_ancestries

    def with_config(self, config: MarketModelConfig) -> Self:
        self.with_config_calls.append(config)
        return type(self)(
            config,
            self.regime_config,
            self.bond_duration,
            self.with_config_calls,
            self.generated_types,
            (*self.ancestry, config),
            self.generated_ancestries,
        )

    def generate(
        self,
        *,
        paths: int,
        horizon: int,
        seed: int,
        antithetic: bool = True,
    ) -> MarketPaths:
        self.generated_types.append(type(self))
        self.generated_ancestries.append(self.ancestry)
        return super().generate(
            paths=paths,
            horizon=horizon,
            seed=seed,
            antithetic=antithetic,
        )


class InvalidSpendingPolicy:
    def __init__(self, invalid_target: str) -> None:
        self.invalid_target = invalid_target

    def target(self, context: SpendingContext) -> np.ndarray:
        wealth = context.wealth
        if self.invalid_target == "wrong_shape":
            return np.array(0.0)
        value = {
            "negative": -1.0,
            "nan": np.nan,
            "infinite": np.inf,
        }[self.invalid_target]
        return np.full_like(wealth, value)


class FirstYearSpending:
    def __init__(self, amount: float) -> None:
        self.amount = amount

    def target(self, context: SpendingContext) -> np.ndarray:
        amount = self.amount if context.year == 0 else 0.0
        return np.full_like(context.wealth, amount)


class YearlyRecoveryIncomeRisk:
    """Test model whose realized fraction identifies each vector transition."""

    def __init__(self) -> None:
        self.transition_path_counts: list[int] = []

    def initial_state(self, paths: int) -> IncomeRiskState:
        return IncomeRiskState(
            displaced=np.zeros(paths, dtype=bool),
            years_since_displacement=np.zeros(paths, dtype=int),
        )

    def transition(
        self,
        context: IncomeRiskContext,
        state: IncomeRiskState,
    ) -> IncomeRiskStep:
        self.transition_path_counts.append(context.deterministic_income.size)
        fraction = 1.0 - 0.10 * len(self.transition_path_counts)
        return IncomeRiskStep(
            state=state,
            realized_income=context.deterministic_income * fraction,
            income_fraction=np.full_like(context.deterministic_income, fraction),
            vesting_eligible=np.ones_like(context.working),
        )


class MalformedPathCountIncomeRisk:
    """Return internally consistent protocol records with the wrong path count."""

    def __init__(self, malformed_stage: str) -> None:
        self.malformed_stage = malformed_stage

    def initial_state(self, paths: int) -> IncomeRiskState:
        output_paths = 1 if self.malformed_stage == "initial state" else paths
        return IncomeRiskState(
            displaced=np.zeros(output_paths, dtype=bool),
            years_since_displacement=np.zeros(output_paths, dtype=int),
        )

    def transition(
        self,
        context: IncomeRiskContext,
        state: IncomeRiskState,
    ) -> IncomeRiskStep:
        output_paths = (
            1
            if self.malformed_stage == "transition step"
            else context.deterministic_income.size
        )
        next_state = IncomeRiskState(
            displaced=np.zeros(output_paths, dtype=bool),
            years_since_displacement=np.zeros(output_paths, dtype=int),
        )
        return IncomeRiskStep(
            state=next_state,
            realized_income=np.full(output_paths, 100.0),
            income_fraction=np.ones(output_paths),
            vesting_eligible=np.ones(output_paths, dtype=bool),
        )


def _two_working_year_scenario() -> PlanningScenario:
    scenario = PlanningScenario()
    return replace(
        scenario,
        person=replace(
            scenario.person,
            current_age=64,
            retirement_age=66,
            maximum_age=66,
        ),
        income=replace(
            scenario.income,
            current_salary=100_000.0,
            defined_contribution=0.0,
            explicit_real_income=(100_000.0, 100_000.0),
            social_security_enabled=False,
        ),
    )


class AllAvailableCashInsurancePolicy:
    def decide(self, context: object) -> RollingDecision:
        spending = context.spending  # type: ignore[attr-defined]
        available = np.maximum(spending.wealth + spending.income, 0.0)
        insurance_price = context.insurance_price  # type: ignore[attr-defined]
        coverage = np.divide(
            available,
            insurance_price,
            out=np.zeros_like(available),
            where=insurance_price > 0,
        )
        paths = len(available)
        return RollingDecision(
            spending=np.zeros(paths),
            equity_fraction=np.zeros(paths),
            leverage=np.ones(paths),
            insured_bequest=coverage,
            insurance_premium=available,
        )


class FixedPremiumInsurancePolicy:
    def __init__(self, premium: float) -> None:
        self.premium = premium

    def decide(self, context: object) -> RollingDecision:
        spending = context.spending  # type: ignore[attr-defined]
        insurance_price = context.insurance_price  # type: ignore[attr-defined]
        existing = context.existing_insured_bequest  # type: ignore[attr-defined]
        paths = len(spending.wealth)
        premium = np.full(paths, self.premium if spending.year == 0 else 0.0)
        return RollingDecision(
            spending=np.zeros(paths),
            equity_fraction=np.zeros(paths),
            leverage=np.ones(paths),
            insured_bequest=existing + premium / insurance_price,
            insurance_premium=premium,
        )


class FixedLeveragePolicy:
    def __init__(self, leverage: float) -> None:
        self.leverage = leverage

    def decide(self, context: object) -> RollingDecision:
        spending = context.spending  # type: ignore[attr-defined]
        existing = context.existing_insured_bequest  # type: ignore[attr-defined]
        paths = len(spending.wealth)
        return RollingDecision(
            spending=np.zeros(paths),
            equity_fraction=np.ones(paths),
            leverage=np.full(paths, self.leverage),
            insured_bequest=existing.copy(),
            insurance_premium=np.zeros(paths),
        )


class ScriptedEquityLossMarket(RegimeSwitchingMarket):
    def generate(
        self,
        *,
        paths: int,
        horizon: int,
        seed: object,
        antithetic: bool = True,
    ) -> MarketPaths:
        del seed, antithetic
        equity = np.zeros((paths, horizon))
        equity[:, 0] = -0.30
        zeros = np.zeros((paths, horizon))
        return MarketPaths(equity, zeros.copy(), zeros.copy(), zeros.copy())


class SeedRecordingMarket(RegimeSwitchingMarket):
    def __init__(self) -> None:
        super().__init__()
        self.seeds: list[object] = []

    def generate(
        self,
        *,
        paths: int,
        horizon: int,
        seed: object,
        antithetic: bool = True,
    ) -> MarketPaths:
        self.seeds.append(seed)
        return super().generate(
            paths=paths,
            horizon=horizon,
            seed=seed,  # type: ignore[arg-type]
            antithetic=antithetic,
        )


class CountingPlanner(LifecyclePlanner):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def plan(self, scenario: PlanningScenario | None = None):  # type: ignore[override]
        self.calls += 1
        return super().plan(scenario)


class RecordingArrayAllocator:
    """Record explicit simulation-module array allocations without replacing NumPy globally."""

    def __init__(self) -> None:
        self.shapes: list[tuple[int, ...]] = []

    def _record(self, shape: int | tuple[int, ...]) -> None:
        self.shapes.append((shape,) if isinstance(shape, int) else tuple(shape))

    def zeros(self, shape: int | tuple[int, ...], *args: object, **kwargs: object) -> np.ndarray:
        self._record(shape)
        return np.zeros(shape, *args, **kwargs)

    def ones(self, shape: int | tuple[int, ...], *args: object, **kwargs: object) -> np.ndarray:
        self._record(shape)
        return np.ones(shape, *args, **kwargs)

    def empty(self, shape: int | tuple[int, ...], *args: object, **kwargs: object) -> np.ndarray:
        self._record(shape)
        return np.empty(shape, *args, **kwargs)

    def full(
        self,
        shape: int | tuple[int, ...],
        fill_value: object,
        *args: object,
        **kwargs: object,
    ) -> np.ndarray:
        self._record(shape)
        return np.full(shape, fill_value, *args, **kwargs)

    def __getattr__(self, name: str) -> object:
        return getattr(np, name)


class ShapeSensitiveLinearCurve(LinearCurve):
    """A custom subclass whose vectorized behavior depends on input rank."""

    def evaluate(self, value: object) -> np.ndarray:
        values = np.asarray(value, dtype=float)
        fill = 0.0 if values.ndim == 2 else 1.0
        return np.full_like(values, fill, dtype=float)


class ScalarDiagnosticLinearCurve(LinearCurve):
    """A custom scalar-outcome curve with a path-independent diagnostic."""

    def diagnostic_breach(self, value: object) -> np.bool_:
        values = np.asarray(value, dtype=float)
        return np.bool_(np.all(values >= 0.0))


def _streaming_utility_addons() -> list[UtilityAddon]:
    return [
        UtilityAddon(
            "spending_last",
            OutcomeType.SPENDING,
            LinearCurve(slope=1e-5),
            aggregation=UtilityAggregation.LAST,
            minimum_age=65,
        ),
        UtilityAddon(
            "insured_bequest_sum",
            OutcomeType.INSURED_BEQUEST,
            LinearCurve(slope=1e-7),
            aggregation=UtilityAggregation.DISCOUNTED_SUM,
            maximum_age=67,
        ),
        UtilityAddon(
            "leverage_worst",
            OutcomeType.LEVERAGE,
            TargetCurve(target=1.25, tolerance=0.1),
            aggregation=UtilityAggregation.WORST,
        ),
        UtilityAddon(
            "equity_mean",
            OutcomeType.ALLOCATION_EQUITY,
            LinearCurve(),
            aggregation=UtilityAggregation.DISCOUNTED_MEAN,
            age_reference=66,
            age_growth=0.02,
        ),
        UtilityAddon(
            "retired_sum",
            OutcomeType.RETIRED,
            LinearCurve(slope=0.5),
            aggregation=UtilityAggregation.DISCOUNTED_SUM,
        ),
        UtilityAddon(
            "working_last",
            OutcomeType.WORKING,
            LinearCurve(slope=0.25),
            aggregation=UtilityAggregation.LAST,
        ),
        UtilityAddon(
            "terminal_wealth",
            OutcomeType.TERMINAL_WEALTH,
            LinearCurve(slope=1e-7),
        ),
    ]


def _short_streaming_scenario(elasticity: float = 0.4) -> PlanningScenario:
    scenario = PlanningScenario()
    return replace(
        scenario,
        person=replace(
            scenario.person,
            current_age=64,
            retirement_age=66,
            maximum_age=68,
            social_security_claim_age=67,
        ),
        preferences=replace(
            scenario.preferences,
            consumption_elasticity=elasticity,
        ),
    )


def test_engine_defaults_to_regime_market() -> None:
    assert isinstance(MonteCarloEngine().market, RegimeSwitchingMarket)


def test_chunked_simulation_prepares_the_plan_once() -> None:
    planner = CountingPlanner()
    engine = MonteCarloEngine(planner=planner)

    engine.simulate_chunked(settings=SimulationSettings(paths=9), chunk_size=4)

    assert planner.calls == 1


@pytest.mark.parametrize("annual_amount", [-1.0, np.nan, np.inf, -np.inf])
def test_fixed_spending_rejects_invalid_annual_amount(annual_amount: float) -> None:
    with pytest.raises(ValueError):
        FixedSpending(annual_amount)


@pytest.mark.parametrize("real_growth", [np.nan, np.inf, -np.inf])
def test_fixed_spending_rejects_nonfinite_real_growth(real_growth: float) -> None:
    with pytest.raises(ValueError):
        FixedSpending(40_000.0, real_growth)


@pytest.mark.parametrize("invalid_target", ["negative", "nan", "infinite", "wrong_shape"])
def test_simulation_rejects_invalid_custom_spending_targets(invalid_target: str) -> None:
    scenario = PlanningScenario()
    scenario = replace(
        scenario,
        person=replace(
            scenario.person,
            current_age=64,
            retirement_age=66,
            maximum_age=66,
        ),
    )

    with pytest.raises(ValueError):
        MonteCarloEngine().simulate(
            scenario,
            settings=SimulationSettings(
                paths=2,
                seed=5,
                stochastic_lifespan=False,
                stochastic_income=False,
            ),
            spending_policy=InvalidSpendingPolicy(invalid_target),
        )


def test_employer_match_is_an_untaxed_working_year_resource() -> None:
    scenario = PlanningScenario()
    scenario = replace(
        scenario,
        person=replace(
            scenario.person,
            current_age=64,
            retirement_age=66,
            maximum_age=66,
        ),
        income=replace(
            scenario.income,
            current_salary=100_000.0,
            defined_contribution=0.0,
            employer_match_rate=0.5,
            explicit_real_income=(100_000.0, 100_000.0),
        ),
    )
    matched = replace(
        scenario,
        income=replace(scenario.income, defined_contribution=10_000.0),
    )
    settings = SimulationSettings(
        paths=2,
        seed=9,
        stochastic_lifespan=False,
        stochastic_income=False,
    )

    without_match = MonteCarloEngine().simulate(scenario, settings=settings)
    with_match = MonteCarloEngine().simulate(matched, settings=settings)

    np.testing.assert_allclose(
        with_match.income_paths - without_match.income_paths,
        np.array([[5_000.0, 5_000.0, 0.0]] * 2),
    )


def test_job_loss_scales_employer_match_with_employment_income() -> None:
    scenario = PlanningScenario()
    scenario = replace(
        scenario,
        person=replace(
            scenario.person,
            current_age=64,
            retirement_age=66,
            maximum_age=66,
        ),
        income=replace(
            scenario.income,
            current_salary=100_000.0,
            defined_contribution=0.0,
            employer_match_rate=0.5,
            explicit_real_income=(100_000.0, 100_000.0),
        ),
    )
    matched = replace(
        scenario,
        income=replace(scenario.income, defined_contribution=10_000.0),
    )
    settings = SimulationSettings(
        paths=32,
        seed=3,
        stochastic_lifespan=False,
        stochastic_income=True,
        job_loss_probability=1.0,
        job_loss_income_fraction=0.25,
    )

    without_match = MonteCarloEngine().simulate(scenario, settings=settings)
    with_match = MonteCarloEngine().simulate(matched, settings=settings)
    working_year_difference = (
        with_match.income_paths[:, :2] - without_match.income_paths[:, :2]
    )

    np.testing.assert_allclose(
        np.unique(working_year_difference),
        np.array([1_250.0, 5_000.0]),
    )


def test_explicit_persistent_income_risk_applies_recovery_fractions_deterministically() -> None:
    scenario = _two_working_year_scenario()
    settings = SimulationSettings(
        paths=4,
        seed=17,
        stochastic_lifespan=False,
    )
    model = PersistentDisplacementIncomeRisk(
        baseline_probability=1.0,
        market_sensitivity=0.0,
        probability_cap=1.0,
        income_fractions_after_displacement=(0.25, 0.70),
    )

    deterministic = MonteCarloEngine().simulate(
        scenario,
        settings=replace(settings, stochastic_income=False),
    )
    first = MonteCarloEngine(income_risk_model=model).simulate(scenario, settings=settings)
    second = MonteCarloEngine(income_risk_model=model).simulate(scenario, settings=settings)

    np.testing.assert_allclose(
        first.income_paths[:, :2],
        deterministic.income_paths[:, :2] * np.array([0.25, 0.70]),
    )
    np.testing.assert_allclose(first.income_paths[:, 2], 0.0)
    np.testing.assert_array_equal(first.income_paths, second.income_paths)


def test_disabled_stochastic_income_ignores_injected_income_risk_model() -> None:
    settings = SimulationSettings(
        paths=4,
        seed=21,
        stochastic_lifespan=False,
        stochastic_income=False,
    )
    model = PersistentDisplacementIncomeRisk(
        baseline_probability=1.0,
        market_sensitivity=0.0,
        probability_cap=1.0,
        income_fractions_after_displacement=(0.0,),
    )

    expected = MonteCarloEngine().simulate(settings=settings)
    actual = MonteCarloEngine(income_risk_model=model).simulate(settings=settings)

    np.testing.assert_array_equal(actual.income_paths, expected.income_paths)


def test_default_stochastic_income_repeats_with_fixed_seed() -> None:
    settings = SimulationSettings(
        paths=8,
        seed=22,
        stochastic_lifespan=False,
        job_loss_probability=0.50,
    )
    engine = MonteCarloEngine()

    first = engine.simulate(settings=settings)
    second = engine.simulate(settings=settings)

    np.testing.assert_array_equal(first.income_paths, second.income_paths)


def test_income_risk_model_receives_one_vector_transition_per_working_year() -> None:
    scenario = _two_working_year_scenario()
    settings = SimulationSettings(
        paths=4,
        seed=23,
        stochastic_lifespan=False,
    )
    model = YearlyRecoveryIncomeRisk()
    deterministic = MonteCarloEngine().simulate(
        scenario,
        settings=replace(settings, stochastic_income=False),
    )

    result = MonteCarloEngine(income_risk_model=model).simulate(scenario, settings=settings)

    assert model.transition_path_counts == [4, 4]
    np.testing.assert_allclose(
        result.income_paths[:, :2],
        deterministic.income_paths[:, :2] * np.array([0.90, 0.80]),
    )


@pytest.mark.parametrize("malformed_stage", ["initial state", "transition step"])
def test_engine_rejects_income_risk_model_output_with_wrong_path_count(
    malformed_stage: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=rf"income risk {malformed_stage} must contain exactly 4 paths",
    ):
        MonteCarloEngine(
            income_risk_model=MalformedPathCountIncomeRisk(malformed_stage)
        ).simulate(
            _two_working_year_scenario(),
            settings=SimulationSettings(
                paths=4,
                seed=29,
                stochastic_lifespan=False,
            ),
        )


def test_simulation_scores_explicit_work_and_retirement_flows() -> None:
    result = MonteCarloEngine(
        utility_addons=[
            UtilityAddon(
                "retirement_freedom",
                OutcomeType.RETIRED,
                LinearCurve(),
                aggregation=UtilityAggregation.DISCOUNTED_SUM,
            ),
            UtilityAddon(
                "work_quality",
                OutcomeType.WORKING,
                LinearCurve(slope=0.1),
                aggregation=UtilityAggregation.DISCOUNTED_SUM,
            ),
        ]
    ).simulate(settings=SimulationSettings(paths=4, seed=8))

    assert np.all(result.utility_component_scores["retirement_freedom"] >= 0.0)
    assert np.all(result.utility_component_scores["work_quality"] >= 0.0)


def test_simulation_is_reproducible_and_limited_liability() -> None:
    engine = MonteCarloEngine()
    settings = SimulationSettings(paths=32, seed=7)
    first = engine.simulate(settings=settings)
    second = engine.simulate(settings=settings)

    np.testing.assert_array_equal(first.wealth_paths, second.wealth_paths)
    np.testing.assert_array_equal(first.spending_paths, second.spending_paths)
    assert np.all(first.wealth_paths >= 0)
    assert first.wealth_paths.shape == (32, PlanningScenario().person.horizon + 1)
    assert first.spending_paths.shape == (32, PlanningScenario().person.horizon)
    assert np.all(np.isfinite(first.certainty_equivalents))
    assert 0 <= first.insolvency_probability <= 1


@pytest.mark.parametrize("instrument", list(LeverageInstrument))
def test_all_leverage_instruments_run(instrument: LeverageInstrument) -> None:
    result = MonteCarloEngine().simulate(
        settings=SimulationSettings(
            paths=8,
            seed=11,
            leverage=2.0,
            leverage_instrument=instrument,
        ),
        allocation_override=Allocation(1.0, 0.0, 0.0, 0.0),
    )
    assert result.wealth_paths.shape[0] == 8
    assert np.all(np.isfinite(result.wealth_paths))
    assert np.all(result.margin_calls >= 0)
    assert np.all(result.decision_paths[OutcomeType.ALLOCATION_EQUITY][:, 0] == pytest.approx(1.0))


def test_public_leverage_return_uses_engine_futures_conventions() -> None:
    market = MarketPaths(
        equity_returns=np.array([[0.10]]),
        bond_returns=np.array([[0.02]]),
        cash_returns=np.array([[0.01]]),
        real_rates=np.array([[0.01]]),
    )
    result = leveraged_portfolio_return(
        market,
        0,
        np.array([1.0]),
        np.array([1.5]),
        LeverageInstrument.FUTURES,
        market_config=MarketModelConfig(),
        leverage_terms=LeverageTerms(
            futures_tax_rate=0.0,
            futures_financing_spread=0.002,
            futures_roll_cost=0.001,
        ),
    )
    assert result[0] == pytest.approx(0.10 + 0.5 * (0.10 - 0.013))

    with pytest.raises(ValueError, match="effective_leverage"):
        leveraged_portfolio_return(
            market,
            0,
            np.array([1.0]),
            np.array([0.5]),
            LeverageInstrument.FUTURES,
            market_config=MarketModelConfig(),
        )


def test_zero_resources_are_counted_as_shortfall() -> None:
    scenario = PlanningScenario()
    scenario = replace(
        scenario,
        wealth=replace(
            scenario.wealth,
            domestic_equity=0,
            global_equity=0,
            bonds=0,
            cash=0,
        ),
        income=replace(
            scenario.income,
            current_salary=0,
            defined_contribution=0,
            social_security_enabled=False,
        ),
    )
    result = MonteCarloEngine().simulate(
        scenario,
        settings=SimulationSettings(paths=4, seed=1),
    )
    assert result.insolvency_probability == 1.0
    assert result.policy_shortfall_probability == 0.0
    np.testing.assert_array_equal(result.certainty_equivalents, np.zeros(4))
    assert result.diagnostics.preferences["spending_floor"].breach_probability == 1.0


def test_accumulator_policy_does_not_treat_future_income_as_current_cash() -> None:
    scenario = PlanningScenario()
    scenario = replace(
        scenario,
        person=replace(
            scenario.person,
            current_age=30,
            retirement_age=67,
            maximum_age=70,
        ),
        wealth=replace(
            scenario.wealth,
            domestic_equity=0.0,
            global_equity=0.0,
            bonds=0.0,
            cash=5_000.0,
        ),
        income=replace(
            scenario.income,
            current_salary=100_000.0,
            defined_contribution=0.0,
        ),
    )

    result = MonteCarloEngine().simulate(
        scenario,
        settings=SimulationSettings(
            paths=64,
            seed=41,
            stochastic_lifespan=False,
            stochastic_income=False,
        ),
    )

    assert result.insolvency_probability == 0.0
    assert result.policy_shortfall_probability == 0.0


def test_custom_spending_is_reserved_before_optimizer_insurance() -> None:
    scenario = PlanningScenario()
    scenario = replace(
        scenario,
        person=replace(
            scenario.person,
            current_age=64,
            retirement_age=65,
            maximum_age=65,
        ),
        wealth=replace(
            scenario.wealth,
            domestic_equity=0.0,
            global_equity=0.0,
            bonds=0.0,
            cash=100_000.0,
        ),
        income=replace(
            scenario.income,
            current_salary=0.0,
            defined_contribution=0.0,
            social_security_enabled=False,
        ),
    )
    result = MonteCarloEngine(
        rolling_decision_policy=AllAvailableCashInsurancePolicy(),
    ).simulate(
        scenario,
        settings=SimulationSettings(
            paths=4,
            seed=43,
            stochastic_lifespan=False,
            stochastic_income=False,
        ),
        spending_policy=FirstYearSpending(20_000.0),
    )

    np.testing.assert_allclose(result.spending_paths[:, 0], 20_000.0)
    assert result.policy_shortfall_probability == 0.0


def test_insurance_premium_and_terminal_benefit_are_both_accounted_for() -> None:
    scenario = PlanningScenario()
    scenario = replace(
        scenario,
        person=replace(
            scenario.person,
            current_age=63,
            retirement_age=64,
            maximum_age=64,
        ),
        wealth=replace(
            scenario.wealth,
            domestic_equity=0.0,
            global_equity=0.0,
            bonds=0.0,
            cash=100_000.0,
        ),
        income=replace(
            scenario.income,
            current_salary=0.0,
            defined_contribution=0.0,
            social_security_enabled=False,
        ),
    )
    premium = 10_000.0
    engine = MonteCarloEngine(
        market=ScriptedEquityLossMarket(),
        rolling_decision_policy=FixedPremiumInsurancePolicy(premium),
    )

    result = engine.simulate(
        scenario,
        settings=SimulationSettings(
            paths=4,
            seed=45,
            stochastic_lifespan=False,
            stochastic_income=False,
        ),
        allocation_override=Allocation(0.0, 0.0, 0.0, 1.0),
    )

    coverage = result.decision_paths[OutcomeType.INSURED_BEQUEST][:, 0]
    np.testing.assert_allclose(
        result.wealth_paths[:, -1],
        100_000.0 - premium + coverage,
    )


def test_margin_call_enforces_the_configured_cooldown_leverage() -> None:
    scenario = PlanningScenario()
    scenario = replace(
        scenario,
        person=replace(
            scenario.person,
            current_age=63,
            retirement_age=64,
            maximum_age=65,
        ),
        wealth=replace(
            scenario.wealth,
            domestic_equity=0.0,
            global_equity=0.0,
            bonds=0.0,
            cash=100_000.0,
        ),
        income=replace(
            scenario.income,
            current_salary=0.0,
            defined_contribution=0.0,
            social_security_enabled=False,
        ),
    )
    engine = MonteCarloEngine(
        market=ScriptedEquityLossMarket(),
        rolling_decision_policy=FixedLeveragePolicy(2.0),
    )

    result = engine.simulate(
        scenario,
        settings=SimulationSettings(
            paths=2,
            seed=46,
            leverage=2.0,
            maintenance_margin=0.40,
            margin_call_leverage=1.0,
            margin_call_cooldown_years=2,
            stochastic_lifespan=False,
            stochastic_income=False,
        ),
        allocation_override=Allocation(1.0, 0.0, 0.0, 0.0),
    )

    np.testing.assert_allclose(
        result.decision_paths[OutcomeType.LEVERAGE],
        [[2.0, 1.0, 1.0], [2.0, 1.0, 1.0]],
    )
    np.testing.assert_array_equal(result.margin_calls, [1, 1])


@pytest.mark.parametrize("instrument", list(LeverageInstrument))
def test_instrument_costs_are_charged_in_the_leverage_decision(
    instrument: LeverageInstrument,
) -> None:
    scenario = PlanningScenario()
    scenario = replace(
        scenario,
        preferences=replace(
            scenario.preferences,
            risk_tolerance=1.0,
            bequest_strength=0.0,
            spending_floor_importance=0.0,
        ),
    )
    market = RegimeSwitchingMarket().with_config_overrides(
        equity_risk_premium=0.10,
        margin_spread=0.10,
    )

    leverage_terms = LeverageTerms(
        futures_tax_rate=0.0,
        futures_financing_spread=0.10,
        futures_roll_cost=0.0,
        box_financing_spread=0.10,
        box_dividend_yield=0.0,
    )

    result = MonteCarloEngine(
        market=market,
        leverage_terms=leverage_terms,
    ).simulate(
        scenario,
        settings=SimulationSettings(
            paths=4,
            seed=47,
            leverage=2.0,
            leverage_instrument=instrument,
            stochastic_lifespan=False,
            stochastic_income=False,
        ),
        allocation_override=Allocation(1.0, 0.0, 0.0, 0.0),
    )

    np.testing.assert_array_equal(
        result.decision_paths[OutcomeType.LEVERAGE][:, 0],
        np.ones(4),
    )


def test_chunked_simulation_passes_full_child_seed_sequences() -> None:
    market = SeedRecordingMarket()

    MonteCarloEngine(market=market).simulate_chunked(
        settings=SimulationSettings(paths=5, seed=53),
        chunk_size=2,
    )

    assert len(market.seeds) == 3
    assert all(isinstance(seed, np.random.SeedSequence) for seed in market.seeds)


def test_sweeps_return_stable_shapes_and_optima() -> None:
    engine = MonteCarloEngine()
    scenario = PlanningScenario()
    settings = SimulationSettings(paths=8, seed=13)
    one = parameter_sweep(
        engine,
        scenario,
        settings,
        parameter="leverage",
        values=[1.0, 1.5],
    )
    assert len(one.metrics) == 2
    assert one.optimum[0] in one.values

    two = parameter_sweep_2d(
        engine,
        scenario,
        settings,
        x_parameter="leverage",
        x_values=[1.0, 1.5],
        y_parameter="risk_tolerance",
        y_values=[0.4, 0.6],
    )
    assert two.metrics.shape == (2, 2)
    assert two.optimum[0] in two.x_values
    assert two.optimum[1] in two.y_values


def test_sweeps_reconfigure_and_retain_the_active_market_type() -> None:
    market = RecordingRegimeMarket()
    engine = MonteCarloEngine(market=market)
    scenario = PlanningScenario()
    settings = SimulationSettings(paths=2, seed=13)

    parameter_sweep(
        engine,
        scenario,
        settings,
        parameter="equity_risk_premium",
        values=[0.03, 0.04],
    )
    assert len(market.with_config_calls) == 2
    assert market.generated_types == [RecordingRegimeMarket, RecordingRegimeMarket]
    assert [len(ancestry) for ancestry in market.generated_ancestries] == [1, 1]

    market.with_config_calls.clear()
    market.generated_types.clear()
    market.generated_ancestries.clear()
    parameter_sweep_2d(
        engine,
        scenario,
        settings,
        x_parameter="equity_volatility",
        x_values=[0.07, 0.08],
        y_parameter="equity_risk_premium",
        y_values=[0.03, 0.04],
    )
    assert len(market.with_config_calls) == 6
    assert market.generated_types == [RecordingRegimeMarket] * 4
    assert [len(ancestry) for ancestry in market.generated_ancestries] == [2, 2, 2, 2]
    assert [
        (
            ancestry[0].equity_risk_premium,
            ancestry[1].equity_risk_premium,
            ancestry[1].equity_volatility,
        )
        for ancestry in market.generated_ancestries
    ] == [
        (0.03, 0.03, 0.07),
        (0.03, 0.03, 0.08),
        (0.04, 0.04, 0.07),
        (0.04, 0.04, 0.08),
    ]


def test_sweep_decisions_are_ranked_by_composite_utility() -> None:
    prefer_moderate_leverage = UtilityAddon(
        name="leverage_preference",
        outcome="leverage",
        curve=TargetCurve(target=1.25, tolerance=0.1),
        importance=100.0,
    )
    engine = MonteCarloEngine(utility_addons=[prefer_moderate_leverage])
    result = parameter_sweep(
        engine,
        PlanningScenario(),
        SimulationSettings(paths=8, seed=4),
        parameter="leverage",
        values=[1.0, 1.25, 1.5],
        metric="median_terminal_wealth",
    )

    assert result.optimum[0] == 1.25
    assert result.metrics != result.utilities


def test_chunked_summary_and_full_results_are_reproducible() -> None:
    engine = MonteCarloEngine()
    settings = SimulationSettings(paths=25, seed=123)

    first = engine.simulate_chunked(
        settings=settings,
        chunk_size=8,
    )
    second = engine.simulate_chunked(
        settings=settings,
        chunk_size=8,
    )
    full = engine.simulate_chunked(
        settings=settings,
        chunk_size=8,
        retain_paths=True,
    )
    repeated_full = engine.simulate_chunked(
        settings=settings,
        chunk_size=8,
        retain_paths=True,
    )

    np.testing.assert_array_equal(first.utility_scores, second.utility_scores)
    np.testing.assert_allclose(
        first.utility_scores,
        full.utility_scores,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        first.diagnostics.insolvent,
        full.diagnostics.insolvent,
    )
    assert first.utility_component_scores.keys() == full.utility_component_scores.keys()
    for name, values in first.utility_component_scores.items():
        np.testing.assert_allclose(
            values,
            full.utility_component_scores[name],
            rtol=1e-12,
            atol=1e-12,
        )
    assert first.diagnostics.preferences.keys() == full.diagnostics.preferences.keys()
    for name, diagnostic in first.diagnostics.preferences.items():
        full_diagnostic = full.diagnostics.preferences[name]
        np.testing.assert_array_equal(diagnostic.breach_count, full_diagnostic.breach_count)
        np.testing.assert_allclose(
            diagnostic.utility_loss,
            full_diagnostic.utility_loss,
            rtol=1e-12,
            atol=1e-12,
        )
    assert full.decision_paths.keys() == repeated_full.decision_paths.keys()
    for outcome, values in full.decision_paths.items():
        np.testing.assert_array_equal(values, repeated_full.decision_paths[outcome])
    assert first.paths == 25
    assert full.wealth_paths.shape == (25, PlanningScenario().person.horizon + 1)


def test_streaming_summary_avoids_annual_simulation_histories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _short_streaming_scenario()
    settings = SimulationSettings(paths=10, seed=123)
    recorder = RecordingArrayAllocator()
    monkeypatch.setattr(simulation_module, "np", recorder)

    MonteCarloEngine(utility_addons=_streaming_utility_addons()).simulate_chunked(
        scenario,
        settings=settings,
        chunk_size=settings.paths,
    )

    annual_shapes = {
        (settings.paths, scenario.person.horizon),
        (settings.paths, scenario.person.horizon + 1),
    }
    assert annual_shapes.isdisjoint(recorder.shapes), recorder.shapes


@pytest.mark.parametrize("elasticity", [0.4, 1.0, 2.0])
def test_streaming_summary_matches_full_utility_outcomes(elasticity: float) -> None:
    scenario = _short_streaming_scenario(elasticity)
    settings = SimulationSettings(paths=10, seed=456, leverage=1.5)
    engine = MonteCarloEngine(utility_addons=_streaming_utility_addons())

    first = engine.simulate_chunked(
        scenario,
        settings=settings,
        chunk_size=settings.paths,
    )
    repeated = engine.simulate_chunked(
        scenario,
        settings=settings,
        chunk_size=settings.paths,
    )
    full = engine.simulate_chunked(
        scenario,
        settings=settings,
        chunk_size=settings.paths,
        retain_paths=True,
    )

    np.testing.assert_array_equal(first.terminal_wealth, repeated.terminal_wealth)
    np.testing.assert_array_equal(first.certainty_equivalents, repeated.certainty_equivalents)
    np.testing.assert_array_equal(first.utility_scores, repeated.utility_scores)
    np.testing.assert_array_equal(first.margin_calls, repeated.margin_calls)
    np.testing.assert_array_equal(
        first.diagnostics.insolvent,
        repeated.diagnostics.insolvent,
    )
    np.testing.assert_array_equal(
        first.diagnostics.policy_shortfall,
        repeated.diagnostics.policy_shortfall,
    )
    assert first.utility_component_scores.keys() == repeated.utility_component_scores.keys()
    for name, values in first.utility_component_scores.items():
        np.testing.assert_array_equal(values, repeated.utility_component_scores[name])
    assert first.diagnostics.preferences.keys() == repeated.diagnostics.preferences.keys()
    for name, diagnostic in first.diagnostics.preferences.items():
        repeated_diagnostic = repeated.diagnostics.preferences[name]
        np.testing.assert_array_equal(
            diagnostic.breach_count,
            repeated_diagnostic.breach_count,
        )
        np.testing.assert_array_equal(
            diagnostic.utility_loss,
            repeated_diagnostic.utility_loss,
        )
    assert first.summary() == repeated.summary()

    np.testing.assert_array_equal(first.terminal_wealth, full.wealth_paths[:, -1])
    np.testing.assert_allclose(
        first.certainty_equivalents,
        full.certainty_equivalents,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        first.utility_scores,
        full.utility_scores,
        rtol=1e-12,
        atol=1e-12,
    )
    assert first.utility_component_scores.keys() == full.utility_component_scores.keys()
    for name, values in first.utility_component_scores.items():
        np.testing.assert_allclose(
            values,
            full.utility_component_scores[name],
            rtol=1e-12,
            atol=1e-12,
        )
    assert first.diagnostics.preferences.keys() == full.diagnostics.preferences.keys()
    for name, diagnostic in first.diagnostics.preferences.items():
        full_diagnostic = full.diagnostics.preferences[name]
        np.testing.assert_array_equal(
            diagnostic.breach_count,
            full_diagnostic.breach_count,
        )
        np.testing.assert_allclose(
            diagnostic.utility_loss,
            full_diagnostic.utility_loss,
            rtol=1e-12,
            atol=1e-12,
        )


def test_streaming_summary_broadcasts_scalar_terminal_diagnostic() -> None:
    scenario = _short_streaming_scenario()
    settings = SimulationSettings(paths=4, seed=654)
    name = "terminal_diagnostic"
    engine = MonteCarloEngine(
        utility_addons=[
            UtilityAddon(
                name,
                OutcomeType.TERMINAL_WEALTH,
                ScalarDiagnosticLinearCurve(slope=1e-7),
            )
        ]
    )

    full = engine.simulate(scenario, settings=settings)
    summary = engine.simulate_chunked(
        scenario,
        settings=settings,
        chunk_size=settings.paths,
    )

    expected = np.full(
        settings.paths,
        int(full.diagnostics.preferences[name].breach_count),
        dtype=np.int64,
    )
    np.testing.assert_array_equal(
        summary.diagnostics.preferences[name].breach_count,
        expected,
    )
    np.testing.assert_allclose(
        summary.diagnostics.preferences[name].utility_loss,
        full.diagnostics.preferences[name].utility_loss,
        rtol=1e-12,
        atol=1e-12,
    )


def test_summary_falls_back_to_full_histories_for_custom_annual_curve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _short_streaming_scenario()
    settings = SimulationSettings(paths=10, seed=789)
    engine = MonteCarloEngine(
        utility_addons=[
            UtilityAddon(
                "shape_sensitive",
                OutcomeType.WORKING,
                ShapeSensitiveLinearCurve(),
                aggregation=UtilityAggregation.DISCOUNTED_SUM,
            )
        ]
    )
    full = engine.simulate_chunked(
        scenario,
        settings=settings,
        chunk_size=settings.paths,
        retain_paths=True,
    )
    recorder = RecordingArrayAllocator()
    monkeypatch.setattr(simulation_module, "np", recorder)

    summary = engine.simulate_chunked(
        scenario,
        settings=settings,
        chunk_size=settings.paths,
    )

    np.testing.assert_array_equal(summary.terminal_wealth, full.wealth_paths[:, -1])
    np.testing.assert_array_equal(summary.certainty_equivalents, full.certainty_equivalents)
    np.testing.assert_array_equal(summary.utility_scores, full.utility_scores)
    assert summary.utility_component_scores.keys() == full.utility_component_scores.keys()
    for name, values in summary.utility_component_scores.items():
        np.testing.assert_array_equal(values, full.utility_component_scores[name])
    assert summary.summary() == full.summary()
    annual_shapes = {
        (settings.paths, scenario.person.horizon),
        (settings.paths, scenario.person.horizon + 1),
    }
    assert annual_shapes.issubset(recorder.shapes), recorder.shapes


def test_streaming_diagnostics_ignore_underflowed_utility_weights() -> None:
    scenario = _short_streaming_scenario()
    scenario = replace(
        scenario,
        preferences=replace(
            scenario.preferences,
            time_preference=1e200,
        ),
    )
    settings = SimulationSettings(
        paths=4,
        seed=321,
        stochastic_lifespan=False,
    )
    engine = MonteCarloEngine(
        utility_addons=[
            UtilityAddon(
                "retirement_diagnostic",
                OutcomeType.RETIRED,
                TargetCurve(target=0.0, tolerance=0.1),
                minimum_age=66,
            )
        ]
    )

    summary = engine.simulate_chunked(
        scenario,
        settings=settings,
        chunk_size=settings.paths,
    )
    full = engine.simulate_chunked(
        scenario,
        settings=settings,
        chunk_size=settings.paths,
        retain_paths=True,
    )

    expected = np.full(settings.paths, 3, dtype=np.int64)
    np.testing.assert_array_equal(
        summary.diagnostics.preferences["retirement_diagnostic"].breach_count,
        expected,
    )
    np.testing.assert_array_equal(
        summary.diagnostics.preferences["retirement_diagnostic"].breach_count,
        full.diagnostics.preferences["retirement_diagnostic"].breach_count,
    )


@pytest.mark.parametrize(
    ("elasticity", "expected"),
    [
        (0.5, np.array([0.0, 160.0])),
        (1.0, np.array([0.0, 200.0])),
        (2.0, np.array([25.0, 225.0])),
    ],
)
def test_streaming_certainty_equivalent_handles_zero_consumption(
    elasticity: float,
    expected: np.ndarray,
) -> None:
    accumulator = simulation_module._CertaintyEquivalentAccumulator(2, elasticity)

    accumulator.ingest(np.array([100.0, 100.0]), np.ones(2))
    accumulator.ingest(np.array([0.0, 400.0]), np.ones(2))

    np.testing.assert_allclose(accumulator.finalize(), expected)


@pytest.mark.parametrize("elasticity", [0.5, 1.0, 2.0])
def test_streaming_certainty_equivalent_returns_zero_without_active_years(
    elasticity: float,
) -> None:
    accumulator = simulation_module._CertaintyEquivalentAccumulator(2, elasticity)

    accumulator.ingest(np.array([0.0, 100.0]), np.zeros(2))

    np.testing.assert_array_equal(accumulator.finalize(), np.zeros(2))
