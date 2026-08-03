from dataclasses import replace

import numpy as np
import pytest

from lifecycle_finance import (
    IsoelasticCurve,
    LifetimeUtility,
    LinearCurve,
    OutcomeType,
    PlanningScenario,
    Preferences,
    SpendingFloorCurve,
    TargetCurve,
    TaxPolicy,
    UtilityAddon,
    UtilityAggregation,
    UtilityCalibrator,
    UtilityModel,
    UtilityOutcome,
)
from lifecycle_finance.taxes import FilingStatus, progressive_tax
from lifecycle_finance.utility import (
    consumption_utility,
    inverse_isoelastic_utility,
    isoelastic_utility,
    vitality,
)


@pytest.mark.parametrize("elasticity", [0.25, 0.4, 1.0, 2.0])
def test_isoelastic_inverse_round_trip(elasticity: float) -> None:
    values = np.array([1.0, 10.0, 100.0])
    recovered = inverse_isoelastic_utility(
        isoelastic_utility(values, elasticity),
        elasticity,
    )
    np.testing.assert_allclose(recovered, values)


def test_vitality_is_one_before_peak_and_declines_after() -> None:
    preferences = PlanningScenario().preferences
    values = vitality(np.array([20, 30, 50, 80]), preferences)
    assert values[0] == 1.0
    assert values[1] == 1.0
    assert np.all(np.diff(values[1:]) < 0)
    assert values[-1] >= preferences.vitality_floor


def test_spending_floor_is_an_inspectable_soft_preference() -> None:
    curve = SpendingFloorCurve(threshold=40_000, scale=10_000)
    values = curve.evaluate([20_000, 30_000, 40_000, 50_000])

    np.testing.assert_allclose(values, [-4.0, -1.0, 0.0, 0.0])
    assert np.all(np.isfinite(values))


def test_consumption_utility_matches_default_spending_addons() -> None:
    scenario = PlanningScenario()
    spending = np.array([20_000.0, 40_000.0, 80_000.0])
    model = UtilityModel.from_scenario(scenario)
    expected = np.zeros_like(spending)

    for addon in model.addons:
        if addon.outcome is OutcomeType.SPENDING:
            expected += addon.importance * addon.curve.evaluate(spending)

    np.testing.assert_allclose(
        consumption_utility(spending, scenario.preferences),
        expected,
    )


def test_consumption_addons_score_working_year_spending() -> None:
    scenario = PlanningScenario()
    model = UtilityModel.from_scenario(scenario)
    ages = (scenario.person.current_age, scenario.person.current_age + 1)
    decisions = {
        OutcomeType.WORKING: np.array([[True, False]]),
        OutcomeType.RETIRED: np.array([[False, True]]),
    }

    def outcome(first_year_spending: float) -> UtilityOutcome:
        return UtilityOutcome(
            spending=np.array([[first_year_spending, 40_000.0]]),
            exposure=np.ones((1, 2)),
            ages=ages,
            terminal_wealth=np.zeros(1),
            decisions=decisions,
        )

    lower = model.decompose(outcome(40_000.0))
    higher = model.decompose(outcome(80_000.0))

    assert higher["consumption"][0] > lower["consumption"][0]
    assert higher["spending_floor"][0] == lower["spending_floor"][0]


@pytest.mark.parametrize(
    "curve",
    [
        LinearCurve(),
        IsoelasticCurve(40_000, 0.4),
        SpendingFloorCurve(40_000, 10_000),
        TargetCurve(50_000, 5_000),
    ],
)
def test_builtin_curves_fill_preallocated_outputs(curve: object) -> None:
    values = np.array([20_000.0, 40_000.0, 60_000.0])
    utility = np.empty_like(values)
    marginal = np.empty_like(values)

    curve.evaluate_into(values, utility)  # type: ignore[attr-defined]
    curve.marginal_utility_into(values, marginal)  # type: ignore[attr-defined]

    np.testing.assert_array_equal(utility, curve.evaluate(values))  # type: ignore[attr-defined]
    np.testing.assert_array_equal(marginal, curve.marginal_utility(values))  # type: ignore[attr-defined]


def test_linear_spending_floor_has_zero_marginal_utility_at_and_above_threshold() -> None:
    curve = SpendingFloorCurve(threshold=30_000, scale=10_000, curvature=1.0)

    np.testing.assert_array_equal(
        curve.marginal_utility([20_000, 30_000, 40_000]),
        [0.0001, 0.0, 0.0],
    )


def test_certainty_equivalent_counts_zero_consumption_years() -> None:
    scenario = PlanningScenario()
    utility = LifetimeUtility(scenario.person, scenario.preferences)

    ruined = utility.certainty_equivalent([50_000.0, 0.0])
    after_death = utility.certainty_equivalent(
        [50_000.0, 0.0],
        survival=[1.0, 0.0],
    )

    assert ruined == 0.0
    assert after_death == pytest.approx(50_000.0)


@pytest.mark.parametrize(
    "survival",
    [
        [1.0],
        [1.0, np.nan],
        [1.0, -0.1],
        [1.0, 1.1],
    ],
)
def test_certainty_equivalent_validates_survival(survival: list[float]) -> None:
    scenario = PlanningScenario()
    utility = LifetimeUtility(scenario.person, scenario.preferences)

    with pytest.raises(ValueError, match="survival"):
        utility.certainty_equivalent([50_000.0, 50_000.0], survival=survival)


def test_positive_power_certainty_equivalent_handles_zero_consumption() -> None:
    scenario = PlanningScenario()
    preferences = replace(
        scenario.preferences,
        consumption_elasticity=2.0,
        time_preference=0.0,
        vitality_peak_age=scenario.person.maximum_age,
        vitality_floor=1.0,
    )
    utility = LifetimeUtility(scenario.person, preferences)

    assert utility.certainty_equivalent([0.0, 0.0]) == 0.0
    assert utility.certainty_equivalent([0.0, 40_000.0]) == pytest.approx(10_000.0)


def test_lifetime_utility_scales_bequest_reference_from_consumption_dollars() -> None:
    scenario = PlanningScenario()
    preferences = replace(
        scenario.preferences,
        bequest_strength=1.0,
        spending_floor_importance=0.0,
    )
    utility = LifetimeUtility(scenario.person, preferences)

    score = utility.score(
        [preferences.consumption_reference],
        bequest=100_000.0,
        bequest_divisor=2.0,
    )

    expected = IsoelasticCurve(
        reference=80_000.0,
        elasticity=preferences.bequest_flexibility,
    ).evaluate([100_000.0])[0]
    assert score == pytest.approx(expected)


def test_isoelastic_lower_bound_is_smooth_and_keeps_low_values_identifiable() -> None:
    curve = IsoelasticCurve(
        reference=2_000_000.0,
        elasticity=0.25,
        minimum_utility=-10.0,
    )

    values = curve.evaluate([0.0, 100_000.0, 500_000.0, 2_000_000.0])
    marginal = curve.marginal_utility([100_000.0, 500_000.0])

    assert values[0] == pytest.approx(-10.0)
    assert values[0] < values[1] < values[2] < values[3]
    assert values[-1] == pytest.approx(0.0)
    assert np.all(marginal > 0.0)
    assert curve.breakpoints() == (2_000_000.0,)
    finite_difference = (
        curve.evaluate([500_001.0])[0] - curve.evaluate([499_999.0])[0]
    ) / 2.0
    assert marginal[1] == pytest.approx(finite_difference, rel=1e-9)


@pytest.mark.parametrize("elasticity", [0.4, 1.0, 2.0])
def test_isoelastic_curve_reaches_lower_bound_for_all_crra_powers(
    elasticity: float,
) -> None:
    curve = IsoelasticCurve(40_000.0, elasticity, minimum_utility=-10.0)

    assert curve.evaluate([0.0])[0] == pytest.approx(-10.0)
    assert curve.evaluate([40_000.0])[0] == pytest.approx(0.0)


def test_default_bequest_values_identify_calibrated_importance() -> None:
    scenario = PlanningScenario()
    model = UtilityModel.from_scenario(scenario)
    years = scenario.person.horizon
    ages = tuple(range(scenario.person.current_age, scenario.person.maximum_age + 1))

    def outcome(spending: float, bequest: float) -> UtilityOutcome:
        return UtilityOutcome(
            spending=np.full((1, years), spending),
            exposure=np.ones((1, years)),
            ages=ages,
            terminal_wealth=np.array([bequest]),
            decisions={OutcomeType.BEQUEST: bequest},
        )

    calibration = UtilityCalibrator(model).calibrate_importance(
        "bequest",
        outcome(60_000.0, 100_000.0),
        outcome(50_000.0, 500_000.0),
    )

    assert calibration.importance > 0.0
    assert calibration.unweighted_addon_difference < 0.0


def test_utility_addon_rejects_nonrepresentable_score_overflow() -> None:
    outcome = UtilityOutcome(
        spending=np.ones((1, 1)),
        exposure=np.ones((1, 1)),
        ages=(45,),
        terminal_wealth=np.zeros(1),
        decisions={},
    )
    addon = UtilityAddon(
        "overflowing_preference",
        OutcomeType.SPENDING,
        LinearCurve(np.finfo(float).max),
        importance=2.0,
        aggregation=UtilityAggregation.DISCOUNTED_SUM,
    )

    with pytest.raises(ValueError, match=r"overflowing_preference.*representable"):
        addon.score(outcome, np.ones((1, 1)))

    profiled = replace(
        addon,
        name="overflowing_age_profile",
        curve=LinearCurve(),
        importance=1.0,
        age_reference=0.0,
        age_growth=20.0,
    )
    with pytest.raises(ValueError, match=r"overflowing_age_profile.*representable"):
        profiled.score(outcome, np.full((1, 1), np.finfo(float).max))


def test_utility_outcome_prefers_decided_bequest_over_terminal_wealth() -> None:
    outcome = UtilityOutcome(
        spending=np.ones((2, 1)),
        exposure=np.ones((2, 1)),
        ages=(45,),
        terminal_wealth=np.array([1_000_000.0, 2_000_000.0]),
        decisions={OutcomeType.BEQUEST: np.array([100_000.0, 500_000.0])},
    )
    fallback = replace(outcome, decisions={})

    np.testing.assert_array_equal(
        outcome.values(OutcomeType.BEQUEST),
        [100_000.0, 500_000.0],
    )
    np.testing.assert_array_equal(
        fallback.values(OutcomeType.BEQUEST),
        fallback.terminal_wealth,
    )


def test_utility_weights_use_actual_age_offsets_on_sparse_grids() -> None:
    scenario = PlanningScenario()
    ages = (scenario.person.current_age, scenario.person.current_age + 2)
    outcome = UtilityOutcome(
        spending=np.ones((1, 2)),
        exposure=np.ones((1, 2)),
        ages=ages,
        terminal_wealth=np.array([0.0]),
        decisions={},
    )
    weights = UtilityModel.from_scenario(scenario).weights(outcome)
    expected_ratio = (
        vitality(np.array([ages[1]]), scenario.preferences)[0]
        / vitality(np.array([ages[0]]), scenario.preferences)[0]
        / (1.0 + scenario.preferences.time_preference) ** 2
    )

    assert weights[0, 1] / weights[0, 0] == pytest.approx(expected_ratio)


def test_utility_weights_accept_an_empty_age_grid() -> None:
    scenario = PlanningScenario()
    outcome = UtilityOutcome(
        spending=np.empty((1, 0)),
        exposure=np.empty((1, 0)),
        ages=(),
        terminal_wealth=np.array([0.0]),
        decisions={},
    )

    assert UtilityModel.from_scenario(scenario).weights(outcome).shape == (1, 0)


def test_linear_curve_has_constant_marginal_utility() -> None:
    curve = LinearCurve(slope=-2.0, intercept=3.0)
    np.testing.assert_array_equal(curve.evaluate([0.0, 1.0, 2.0]), [3.0, 1.0, -1.0])
    np.testing.assert_array_equal(curve.marginal_utility([0.0, 1.0]), [-2.0, -2.0])
    assert curve.breakpoints() == ()


def test_retirement_is_an_explicit_age_weighted_flow() -> None:
    base = PlanningScenario()
    scenario = replace(
        base,
        preferences=replace(
            base.preferences,
            bequest_strength=0.0,
            spending_floor_importance=0.0,
        ),
    )
    ages = tuple(range(scenario.person.current_age, scenario.person.maximum_age + 1))
    age_values = np.asarray(ages)
    spending = np.full((1, len(ages)), 70_000.0)
    freedom = UtilityAddon(
        name="retirement_freedom",
        outcome=OutcomeType.RETIRED,
        curve=LinearCurve(),
        importance=0.5,
        aggregation=UtilityAggregation.DISCOUNTED_SUM,
        age_reference=40.0,
        age_growth=np.log(1.5) / 20.0,
    )
    model = UtilityModel.from_scenario(scenario, [freedom])

    def outcome(retirement_age: int) -> UtilityOutcome:
        retired = (age_values >= retirement_age)[np.newaxis, :]
        return UtilityOutcome(
            spending=spending,
            exposure=np.ones_like(spending),
            ages=ages,
            terminal_wealth=np.array([0.0]),
            decisions={
                OutcomeType.RETIRED: retired,
                OutcomeType.WORKING: ~retired,
            },
        )

    early = model.decompose(outcome(60))["retirement_freedom"][0]
    late = model.decompose(outcome(65))["retirement_freedom"][0]
    assert early > late > 0.0


def test_risk_tolerance_defaults_to_crra_consumption_elasticity() -> None:
    preferences = Preferences(consumption_elasticity=0.4)
    assert preferences.risk_tolerance is None
    assert preferences.effective_risk_tolerance == pytest.approx(0.4)
    assert replace(preferences, risk_tolerance=0.6).effective_risk_tolerance == pytest.approx(0.6)


def test_legacy_retirement_multiplier_cannot_encode_preferences() -> None:
    with pytest.raises(ValueError, match="RETIRED or WORKING"):
        Preferences(retirement_utility_multiplier=1.2)


def test_utility_model_decomposes_base_and_addon_preferences() -> None:
    scenario = PlanningScenario()
    prefer_retirement_at_65 = UtilityAddon(
        name="retirement_timing",
        outcome="retirement_age",
        curve=TargetCurve(target=65, tolerance=2),
        importance=3,
    )
    model = UtilityModel.from_scenario(scenario, [prefer_retirement_at_65])
    outcome = UtilityOutcome(
        spending=np.full((2, scenario.person.horizon), 50_000.0),
        exposure=np.ones((2, scenario.person.horizon)),
        ages=tuple(range(scenario.person.current_age, scenario.person.maximum_age + 1)),
        terminal_wealth=np.array([1_000_000.0, 1_000_000.0]),
        decisions={"retirement_age": np.array([65.0, 69.0])},
    )

    components = model.decompose(outcome)

    assert set(components) == {
        "consumption",
        "spending_floor",
        "bequest",
        "retirement_timing",
    }
    assert components["retirement_timing"][0] == 0
    assert components["retirement_timing"][1] == pytest.approx(-12)


def test_progressive_tax_uses_first_bracket_rate() -> None:
    brackets = ((10_000.0, 0.10), (20_000.0, 0.20), (float("inf"), 0.30))
    assert progressive_tax(5_000, brackets) == 500
    assert progressive_tax(15_000, brackets) == 2_000


def test_progressive_tax_applies_last_custom_rate_above_final_threshold() -> None:
    assert progressive_tax(200_000, ((50_000.0, 0.10),)) == pytest.approx(20_000)


def test_capital_gains_tax_applies_last_custom_rate_above_final_threshold() -> None:
    policy = TaxPolicy(
        ordinary_brackets=((float("inf"), 0.0),),
        capital_gains_brackets=((50_000.0, 0.10),),
        standard_deduction=0.0,
        state_ordinary_rate=0.0,
        niit_rate=0.0,
    )

    result = policy.calculate(long_term_capital_gains=200_000, include_payroll=False)

    assert result.federal_capital_gains == pytest.approx(20_000)


def test_tax_policy_preserves_an_explicit_zero_standard_deduction() -> None:
    policy = TaxPolicy(standard_deduction=0.0)

    assert policy.standard_deduction == 0.0
    assert policy.calculate(
        ordinary_income=1_000,
        include_payroll=False,
    ).federal_ordinary == pytest.approx(100.0)


def test_2026_tax_policy_applies_deduction_and_capital_gain_stacking() -> None:
    policy = TaxPolicy.for_2026(FilingStatus.SINGLE)
    low_wage = policy.calculate(wages=10_000, include_payroll=False)
    assert low_wage.federal_ordinary == 0

    taxable_wage = policy.calculate(wages=20_000, include_payroll=False)
    assert taxable_wage.federal_ordinary == pytest.approx(390.0)

    gains = policy.calculate(
        ordinary_income=40_000,
        long_term_capital_gains=30_000,
        include_payroll=False,
    )
    assert gains.federal_capital_gains > 0


def test_niit_includes_the_usable_net_capital_loss_deduction() -> None:
    policy = TaxPolicy.for_2026()

    result = policy.calculate(
        wages=400_000,
        nonqualified_dividends=100_000,
        short_term_capital_gains=-3_000,
    )
    loss_exceeds_investment_income = policy.calculate(
        wages=400_000,
        nonqualified_dividends=1_000,
        short_term_capital_gains=-3_000,
    )

    assert result.net_investment_income_tax == pytest.approx(3_686)
    assert loss_exceeds_investment_income.net_investment_income_tax == 0


def test_married_tax_policy_preserves_explicit_income_thresholds() -> None:
    policy = TaxPolicy(
        filing_status=FilingStatus.MARRIED_JOINT,
        niit_threshold=123_456.0,
        additional_medicare_threshold=234_567.0,
    )

    assert policy.niit_threshold == 123_456.0
    assert policy.additional_medicare_threshold == 234_567.0


def test_married_tax_policy_defaults_thresholds_with_custom_brackets() -> None:
    brackets = ((float("inf"), 0.10),)
    policy = TaxPolicy(
        filing_status=FilingStatus.MARRIED_JOINT,
        ordinary_brackets=brackets,
        capital_gains_brackets=brackets,
    )

    assert policy.niit_threshold == 250_000.0
    assert policy.additional_medicare_threshold == 250_000.0


def test_social_security_tax_is_bounded_at_85_percent() -> None:
    policy = TaxPolicy.for_2026()
    taxable = policy.taxable_social_security(
        benefits=40_000,
        other_ordinary_income=200_000,
        investment_income=0,
    )
    assert taxable == pytest.approx(34_000)


def test_social_security_provisional_income_counts_interest_once() -> None:
    result = TaxPolicy.for_2026().calculate(
        interest=10_000,
        social_security=20_000,
        include_payroll=False,
    )

    assert result.taxable_social_security == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (field, value)
        for field in (
            "wages",
            "ordinary_income",
            "qualified_dividends",
            "nonqualified_dividends",
            "interest",
            "social_security",
        )
        for value in (-1.0, float("nan"), float("inf"), float("-inf"))
    ],
)
def test_tax_calculation_rejects_negative_or_nonfinite_income(
    field: str,
    value: float,
) -> None:
    with pytest.raises(ValueError, match=field):
        TaxPolicy.for_2026().calculate(**{field: value})


@pytest.mark.parametrize("field", ["short_term_capital_gains", "long_term_capital_gains"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_tax_calculation_rejects_nonfinite_capital_gains(
    field: str,
    invalid: float,
) -> None:
    with pytest.raises(ValueError, match=field):
        TaxPolicy.for_2026().calculate(**{field: invalid})
