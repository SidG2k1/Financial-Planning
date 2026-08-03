import numpy as np
import pytest

import lifecycle_finance as package
import lifecycle_finance.spending as spending_module
from lifecycle_finance import (
    Allocation,
    AllocationConstraint,
    AmortizedSpending,
    FinancialWealth,
    FixedSpending,
    GompertzMortality,
    IncomePlan,
    LifecyclePlanner,
    LifetimeUtility,
    MarginalUtilitySpending,
    Person,
    PlanningScenario,
    PlanSpending,
    Preferences,
    SimulationSettings,
    SocialSecurityPolicy,
    StaticPriceProvider,
    TaxPolicy,
    UtilityAddonConfig,
    UtilityModel,
    UtilityOptimizedSpending,
)
from lifecycle_finance.cli import main
from lifecycle_finance.income import WorkbookSalaryModel, combined_income_path
from lifecycle_finance.spending import SpendingContext
from lifecycle_finance.taxes import FilingStatus


def test_policy_path_api_is_exported_from_package_root() -> None:
    expected_exports = {
        "PolicyPathContext",
        "PolicyPathDecision",
        "PolicyPathEvaluator",
        "PolicyPathPolicy",
        "PolicyPathResult",
    }

    assert expected_exports <= set(package.__all__)
    for name in expected_exports:
        assert hasattr(package, name)


def test_income_risk_api_is_exported_from_package_root() -> None:
    expected = {
        "IncomeRiskContext",
        "IncomeRiskModel",
        "IncomeRiskPaths",
        "IncomeRiskState",
        "IncomeRiskStep",
        "PersistentDisplacementIncomeRisk",
        "TransitoryMarketJobLoss",
        "generate_income_risk_paths",
    }

    assert expected <= set(package.__all__)
    for name in expected:
        assert hasattr(package, name)


@pytest.mark.parametrize("horizon", [1, 10, 73])
def test_discounted_annuity_and_income_matches_matrix_formula(horizon: int) -> None:
    real_rate = np.array([0.04, 0.0, -0.02])
    conditional_survival = np.linspace(1.0, 0.15, horizon)
    future_income = np.linspace(80_000.0, 20_000.0, horizon)
    current_income = np.array([50_000.0, 60_000.0, 70_000.0])
    offsets = np.arange(horizon)
    discount = np.power(
        1.0 + real_rate[:, np.newaxis],
        -offsets[np.newaxis, :],
    )
    expected_annuity = np.sum(
        discount[:, 1:] * conditional_survival[np.newaxis, 1:],
        axis=1,
    )
    expected_income = current_income + np.sum(
        discount[:, 1:]
        * conditional_survival[np.newaxis, 1:]
        * future_income[np.newaxis, 1:],
        axis=1,
    )

    annuity, income = spending_module._discounted_annuity_and_income(
        real_rate,
        conditional_survival,
        future_income,
        current_income,
    )

    np.testing.assert_allclose(annuity, expected_annuity, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(income, expected_income, rtol=1e-12, atol=0.0)


def test_domain_validation_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        Person(current_age=70, retirement_age=65)
    with pytest.raises(ValueError):
        SimulationSettings(leverage=0.5)


NONFINITE_VALUES = (float("nan"), float("inf"), float("-inf"))


@pytest.mark.parametrize("invalid", NONFINITE_VALUES)
def test_person_rejects_nonfinite_longevity_adjustment(invalid: float) -> None:
    with pytest.raises(ValueError):
        Person(longevity_adjustment=invalid)


@pytest.mark.parametrize(
    "field",
    [
        "current_age",
        "retirement_age",
        "maximum_age",
        "current_year",
        "social_security_claim_age",
    ],
)
@pytest.mark.parametrize("invalid", NONFINITE_VALUES)
def test_person_rejects_nonfinite_age_and_year_fields(
    field: str,
    invalid: float,
) -> None:
    with pytest.raises(ValueError):
        Person(**{field: invalid})


def test_person_allows_finite_negative_longevity_adjustment() -> None:
    assert Person(longevity_adjustment=-2.5).longevity_adjustment == -2.5


@pytest.mark.parametrize(
    "field",
    ["domestic_equity", "global_equity", "bonds", "cash"],
)
@pytest.mark.parametrize("invalid", NONFINITE_VALUES)
def test_financial_wealth_rejects_nonfinite_fields(field: str, invalid: float) -> None:
    with pytest.raises(ValueError):
        FinancialWealth(**{field: invalid})


@pytest.mark.parametrize(
    "field",
    [
        "current_salary",
        "defined_contribution",
        "employer_match_rate",
        "social_security_taxable_max",
    ],
)
@pytest.mark.parametrize("invalid", NONFINITE_VALUES)
def test_income_plan_rejects_nonfinite_scalar_fields(field: str, invalid: float) -> None:
    with pytest.raises(ValueError):
        IncomePlan(**{field: invalid})


@pytest.mark.parametrize("invalid", NONFINITE_VALUES)
def test_income_plan_rejects_nonfinite_explicit_income(invalid: float) -> None:
    with pytest.raises(ValueError):
        IncomePlan(explicit_real_income=(100_000.0, invalid))


@pytest.mark.parametrize(
    "field",
    [
        "time_preference",
        "consumption_elasticity",
        "consumption_reference",
        "risk_tolerance",
        "spending_floor",
        "spending_floor_importance",
        "spending_floor_scale",
        "nondiscretionary_consumption",
        "annuitization_fraction",
        "bequest_flexibility",
        "bequest_strength",
        "fixed_bequest",
        "vitality_peak_age",
        "vitality_half_life",
        "vitality_floor",
        "retirement_utility_multiplier",
    ],
)
@pytest.mark.parametrize("invalid", NONFINITE_VALUES)
def test_preferences_reject_nonfinite_scalar_fields(field: str, invalid: float) -> None:
    with pytest.raises(ValueError):
        Preferences(**{field: invalid})


@pytest.mark.parametrize(
    "field",
    [
        "importance",
        "minimum_age",
        "maximum_age",
        "age_reference",
        "age_growth",
    ],
)
@pytest.mark.parametrize("invalid", NONFINITE_VALUES)
def test_utility_addon_rejects_nonfinite_scalar_fields(
    field: str,
    invalid: float,
) -> None:
    with pytest.raises(ValueError):
        UtilityAddonConfig(
            name="linear spending",
            outcome="spending",
            curve="linear",
            parameters={"slope": 1.0},
            **{field: invalid},
        )


@pytest.mark.parametrize("invalid", NONFINITE_VALUES)
def test_utility_addon_rejects_nonfinite_parameters(invalid: float) -> None:
    with pytest.raises(ValueError):
        UtilityAddonConfig(
            name="linear spending",
            outcome="spending",
            curve="linear",
            parameters={"slope": invalid},
        )


@pytest.mark.parametrize(
    "field",
    [
        "paths",
        "seed",
        "leverage",
        "maintenance_margin",
        "margin_call_leverage",
        "margin_call_cooldown_years",
        "job_loss_probability",
        "job_loss_market_sensitivity",
        "job_loss_income_fraction",
    ],
)
@pytest.mark.parametrize("invalid", NONFINITE_VALUES)
def test_simulation_settings_reject_nonfinite_scalar_fields(
    field: str,
    invalid: float,
) -> None:
    with pytest.raises(ValueError):
        SimulationSettings(**{field: invalid})


@pytest.mark.parametrize(
    ("retirement_age", "expected_claiming_age"),
    [(55, 62), (62, 62), (66, 66), (70, 70), (75, 70)],
)
def test_default_social_security_claiming_age_stays_within_policy_bounds(
    retirement_age: int,
    expected_claiming_age: int,
) -> None:
    person = Person(
        current_age=45,
        retirement_age=retirement_age,
        maximum_age=90,
    )

    assert person.claiming_age == expected_claiming_age


def test_bounded_allocation_projection() -> None:
    constraint = AllocationConstraint(
        lower=Allocation(0.1, 0.1, 0.0, 0.1),
        upper=Allocation(0.5, 0.4, 0.4, 0.7),
    )
    result = constraint.apply(Allocation(0.9, -0.2, 0.1, 0.2))
    assert result.total == pytest.approx(1.0)
    assert 0.1 <= result.domestic_equity <= 0.5
    assert 0.1 <= result.global_equity <= 0.4
    assert 0.1 <= result.cash <= 0.7


def test_mortality_sampling_and_actuarial_prices() -> None:
    mortality = GompertzMortality.from_person(PlanningScenario().person)
    samples = mortality.sample_death_ages(100, np.random.default_rng(1))
    assert samples.min() > mortality.current_age
    assert samples.max() <= mortality.maximum_age + 1
    assert mortality.life_annuity_factor(0.02) > 1
    assert 0 < mortality.term_insurance_price(0, 0.02) < 1
    assert 0 < mortality.permanent_insurance_price(0, 0.02) < 1


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("current_age", np.nan),
        ("modal_age", np.inf),
        ("dispersion", -np.inf),
        ("maximum_age", np.nan),
    ],
)
def test_gompertz_mortality_rejects_nonfinite_parameters(
    field: str,
    invalid: float,
) -> None:
    inputs = {
        "current_age": 45,
        "modal_age": 94.0,
        "dispersion": 9.0,
        "maximum_age": 117,
    }
    inputs[field] = invalid

    with pytest.raises(ValueError, match=field):
        GompertzMortality(**inputs)  # type: ignore[arg-type]


@pytest.mark.parametrize("dispersion", [0.0, -1.0])
def test_gompertz_mortality_requires_positive_dispersion(dispersion: float) -> None:
    with pytest.raises(ValueError, match="dispersion"):
        GompertzMortality(45, 94.0, dispersion, 117)


@pytest.mark.parametrize(
    ("current_age", "maximum_age"),
    [(-1, 117), (45, 44)],
)
def test_gompertz_mortality_requires_ordered_ages(
    current_age: int,
    maximum_age: int,
) -> None:
    with pytest.raises(ValueError, match="age"):
        GompertzMortality(current_age, 94.0, 9.0, maximum_age)


@pytest.mark.parametrize("method", ["term_insurance_price", "permanent_insurance_price"])
@pytest.mark.parametrize("year", [-1, 73, 0.5, np.nan])
def test_insurance_prices_reject_invalid_year(method: str, year: float) -> None:
    mortality = GompertzMortality(45, 94.0, 9.0, 117)

    with pytest.raises(ValueError, match="year"):
        getattr(mortality, method)(year, 0.02)


@pytest.mark.parametrize("method", ["term_insurance_price", "permanent_insurance_price"])
@pytest.mark.parametrize("real_rate", [-1.0, -1.1, np.nan, np.inf, -np.inf])
def test_insurance_prices_reject_invalid_real_rate(
    method: str,
    real_rate: float,
) -> None:
    mortality = GompertzMortality(45, 94.0, 9.0, 117)

    with pytest.raises(ValueError, match="real_rate"):
        getattr(mortality, method)(0, real_rate)


def test_modern_social_security_claiming_adjustments() -> None:
    policy = SocialSecurityPolicy()
    person = Person(
        current_age=45,
        retirement_age=67,
        social_security_claim_age=67,
    )
    earnings = [100_000.0] * 35
    benefit = policy.annual_benefit(person, earnings)
    early = policy.claiming_adjustment(62, 67)
    delayed = policy.claiming_adjustment(70, 67)
    assert benefit > 0
    assert early == pytest.approx(0.70)
    assert delayed == pytest.approx(1.24)


def test_explicit_zero_income_is_the_modern_social_security_earnings_history() -> None:
    person = Person(
        current_age=45,
        retirement_age=50,
        maximum_age=70,
        social_security_claim_age=62,
    )
    income = IncomePlan(
        current_salary=100_000.0,
        explicit_real_income=(0.0, 0.0, 0.0, 0.0, 0.0),
    )

    _, benefit = combined_income_path(
        person,
        income,
        WorkbookSalaryModel(),
        SocialSecurityPolicy(),
    )

    assert benefit == 0.0


def test_explicit_social_security_history_uses_projected_fill_and_truncation() -> None:
    person = Person(
        current_age=45,
        retirement_age=50,
        maximum_age=70,
        social_security_claim_age=62,
    )
    income = IncomePlan(
        explicit_real_income=(10_000.0, 20_000.0),
    )
    policy = SocialSecurityPolicy()
    expected = policy.annual_benefit(
        person,
        [10_000.0, 20_000.0, 20_000.0, 20_000.0, 20_000.0],
    )

    _, benefit = combined_income_path(
        person,
        income,
        WorkbookSalaryModel(),
        policy,
    )

    assert benefit == pytest.approx(expected)


def test_explicit_income_beyond_retirement_year_is_not_dropped() -> None:
    person = Person(current_age=60, retirement_age=65, maximum_age=70)
    income = IncomePlan(
        explicit_real_income=(
            100_000.0,
            100_000.0,
            100_000.0,
            100_000.0,
            100_000.0,
            50_000.0,
            40_000.0,
            30_000.0,
            20_000.0,
            10_000.0,
        ),
    )

    path = WorkbookSalaryModel().project(person, income)

    np.testing.assert_allclose(
        path,
        [
            100_000.0,
            100_000.0,
            100_000.0,
            100_000.0,
            100_000.0,
            50_000.0,
            40_000.0,
            30_000.0,
            20_000.0,
            10_000.0,
            0.0,
        ],
    )


def test_short_explicit_income_still_forward_fills_to_retirement() -> None:
    person = Person(current_age=60, retirement_age=65, maximum_age=70)
    income = IncomePlan(explicit_real_income=(100_000.0, 120_000.0))

    path = WorkbookSalaryModel().project(person, income)

    np.testing.assert_allclose(
        path,
        [100_000.0, 120_000.0, 120_000.0, 120_000.0, 120_000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )


@pytest.mark.parametrize(
    ("claiming_age", "expected_adjustment"),
    [
        (66 + 1 / 12, 1 + 2 / 3 / 100),
        (70, 1.32),
        (70 + 1 / 12, 1.32),
        (71, 1.32),
    ],
)
def test_delayed_retirement_credits_cover_every_month_until_age_70(
    claiming_age: float,
    expected_adjustment: float,
) -> None:
    assert SocialSecurityPolicy.claiming_adjustment(claiming_age, 66) == pytest.approx(
        expected_adjustment
    )


def test_spending_policies_return_vectorized_targets() -> None:
    scenario = PlanningScenario()
    plan = LifecyclePlanner().plan(scenario)
    context = SpendingContext(
        year=0,
        wealth=np.array([1_000_000.0, 2_000_000.0]),
        income=np.array([100_000.0, 100_000.0]),
        future_income=np.asarray(plan.income_path),
        real_rate=np.array([0.02, 0.03]),
        scenario=scenario,
        lifecycle_plan=plan,
    )
    policies = [
        FixedSpending(50_000),
        PlanSpending(),
        AmortizedSpending(),
        MarginalUtilitySpending(),
        UtilityOptimizedSpending(UtilityModel.from_scenario(scenario)),
    ]
    for policy in policies:
        target = policy.target(context)
        assert target.shape == (2,)
        assert np.all(np.isfinite(target))
        assert np.all(target >= 0)


@pytest.mark.parametrize("policy", [AmortizedSpending(), MarginalUtilitySpending()])
def test_analytic_spending_uses_realized_current_income(
    policy: AmortizedSpending | MarginalUtilitySpending,
) -> None:
    scenario = PlanningScenario()
    plan = LifecyclePlanner().plan(scenario)
    common = dict(
        year=0,
        wealth=np.array([1_000_000.0]),
        future_income=np.asarray(plan.income_path),
        real_rate=np.array([0.02]),
        scenario=scenario,
        lifecycle_plan=plan,
    )
    low_income = policy.target(
        SpendingContext(income=np.array([0.0]), **common),
    )
    high_income = policy.target(
        SpendingContext(income=np.array([200_000.0]), **common),
    )

    assert high_income[0] > low_income[0]


def test_marginal_utility_spending_preserves_budget_when_normalized_annuity_is_small() -> None:
    scenario = PlanningScenario(
        person=Person(current_age=20, retirement_age=66, maximum_age=117),
        income=IncomePlan(
            current_salary=0.0,
            defined_contribution=0.0,
            social_security_enabled=False,
        ),
        preferences=Preferences(
            time_preference=0.02,
            consumption_elasticity=3.0,
            vitality_peak_age=117,
            vitality_floor=1.0,
            spending_floor=0.0,
            spending_floor_importance=0.0,
            bequest_strength=0.0,
        ),
    )
    plan = LifecyclePlanner().plan(scenario)
    context = SpendingContext(
        year=0,
        wealth=np.array([1.0]),
        income=np.array([0.0]),
        future_income=np.asarray(plan.income_path),
        real_rate=np.array([0.08]),
        scenario=scenario,
        lifecycle_plan=plan,
    )

    target = MarginalUtilitySpending().target(context)

    assert target[0] == pytest.approx(0.00023176838617836585)


def test_lifetime_utility_scores_and_recovers_ce() -> None:
    scenario = PlanningScenario()
    utility = LifetimeUtility(scenario.person, scenario.preferences)
    spending = np.full(scenario.person.horizon, 75_000.0)
    score = utility.score(spending, bequest=1_000_000.0)
    zero_score = utility.score(np.zeros(scenario.person.horizon), bequest=0.0)
    ce = utility.certainty_equivalent(spending)
    assert np.isfinite(score)
    assert np.isfinite(zero_score)
    assert ce == pytest.approx(75_000)


def test_married_tax_policy_and_static_price_errors() -> None:
    married = TaxPolicy.for_2026(FilingStatus.MARRIED_JOINT)
    assert married.standard_deduction == 32_200
    assert married.calculate(wages=20_000, include_payroll=False).federal_ordinary == 0
    with pytest.raises(KeyError):
        StaticPriceProvider({"A": 1.0}).prices(("A", "B"))


@pytest.mark.parametrize(
    "price",
    [0.0, -1.0, float("nan"), float("inf"), float("-inf")],
)
def test_static_prices_reject_nonpositive_or_nonfinite_values(price: float) -> None:
    with pytest.raises(ValueError, match="positive"):
        StaticPriceProvider({"A": price}).prices(("A",))


def test_cli_example_and_sweep(capsys) -> None:
    assert main(["example"]) == 0
    assert '"current_age": 45' in capsys.readouterr().out
    assert main(["sweep", "leverage", "1", "1.25", "--paths", "4"]) == 0
    sweep_output = capsys.readouterr().out
    assert '"metrics"' in sweep_output
    assert '"optimum"' in sweep_output
