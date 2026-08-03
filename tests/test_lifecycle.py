from pathlib import Path

import numpy as np
import pytest

import lifecycle_finance as package
from lifecycle_finance import (
    BequestMode,
    CapitalMarketAssumptions,
    GompertzMortality,
    LifecyclePlanner,
    PlanningScenario,
    WorkbookSocialInsurance,
)
from lifecycle_finance.demographics import _permanent_insurance_price_curve
from lifecycle_finance.income import WorkbookSalaryModel
from lifecycle_finance.serialization import load_scenario

WORKBOOK_SCENARIO_PATH = Path(__file__).parents[1] / "examples" / "workbook_scenario.json"


def test_workbook_survival_parity() -> None:
    scenario = PlanningScenario()
    adjusted = GompertzMortality.from_person(scenario.person)
    unadjusted = GompertzMortality.from_person(scenario.person, adjusted=False)

    assert adjusted.survival(0, 0) == 1.0
    assert adjusted.survival(0, 1) == pytest.approx(0.9995207151329708)
    assert unadjusted.survival(0, 1) == pytest.approx(0.9993282513741857)
    assert adjusted.survival(0, scenario.person.horizon) == 0.0
    assert np.all(np.diff(adjusted.survival_curve()) <= 0)


def test_permanent_insurance_price_curve_matches_direct_prices() -> None:
    mortality = GompertzMortality.from_person(PlanningScenario().person)
    rate = 0.02
    expected = np.array(
        [
            sum(
                (
                    mortality.survival(year, death_year - 1)
                    - mortality.survival(year, death_year)
                )
                / (1.0 + rate) ** (death_year - year)
                for death_year in range(
                    year + 1,
                    mortality.maximum_age - mortality.current_age + 2,
                )
            )
            for year in range(mortality.maximum_age - mortality.current_age + 1)
        ]
    )

    np.testing.assert_allclose(
        mortality.permanent_insurance_price_curve(rate),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_permanent_insurance_price_curve_returns_copy_of_cached_prices() -> None:
    mortality = GompertzMortality.from_person(PlanningScenario().person)
    rate = 0.02
    expected_first_price = mortality.permanent_insurance_price(0, rate)

    prices = mortality.permanent_insurance_price_curve(rate)
    prices[0] = -1.0

    assert mortality.permanent_insurance_price_curve(rate)[0] == expected_first_price


def test_permanent_insurance_price_builds_curve_once_for_all_years(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _permanent_insurance_price_curve.cache_clear()
    original_survival = GompertzMortality.survival
    calls = 0

    def counted_survival(
        self: GompertzMortality,
        from_year: int | float,
        to_year: int | float,
    ) -> float:
        nonlocal calls
        calls += 1
        return original_survival(self, from_year, to_year)

    monkeypatch.setattr(GompertzMortality, "survival", counted_survival)
    mortality = GompertzMortality(45, 94.0, 8.8840011008432, 117)
    prices = [
        mortality.permanent_insurance_price(year, 0.02)
        for year in range(mortality.maximum_age - mortality.current_age + 1)
    ]

    assert all(price > 0.0 for price in prices)
    assert calls == mortality.maximum_age - mortality.current_age + 1
    _permanent_insurance_price_curve.cache_clear()


def test_workbook_income_parity() -> None:
    scenario = PlanningScenario()
    salary = WorkbookSalaryModel()
    path = salary.project(scenario.person, scenario.income)
    history = salary.backfilled_earnings(scenario.person, scenario.income)
    benefit = WorkbookSocialInsurance().annual_benefit(scenario.person, history)

    assert path[0] == pytest.approx(150_000.0)
    assert path[1] == pytest.approx(150_531.6364843264)
    assert path[scenario.person.retirement_year] == 0.0
    assert benefit == pytest.approx(31_227.616)


def test_workbook_capital_market_parity() -> None:
    scenario = PlanningScenario()
    cma = CapitalMarketAssumptions.workbook_defaults()
    human_mix = cma.asset_mix(scenario.human_capital_exposure)
    liability_mix = cma.asset_mix(scenario.liability_exposure)

    assert cma.reference_weights.sum() == pytest.approx(1.0)
    assert cma.global_equity_fraction == pytest.approx(0.5271385149804603)
    assert cma.equilibrium_discount_rate(human_mix) == pytest.approx(0.03185152652475446, abs=2e-10)
    assert cma.equilibrium_discount_rate(liability_mix) == pytest.approx(
        0.030722490496767698, abs=2e-10
    )


def test_equilibrium_discount_rate_rejects_infeasible_negative_covariance() -> None:
    scenario = PlanningScenario()
    cma = CapitalMarketAssumptions.workbook_defaults()
    human_mix = cma.asset_mix(scenario.human_capital_exposure)

    with pytest.raises(ValueError):
        cma.equilibrium_discount_rate(-60.0 * human_mix)


def test_shipped_workbook_scenario_reproduces_balance_sheet_and_allocation() -> None:
    scenario = load_scenario(WORKBOOK_SCENARIO_PATH)
    plan = LifecyclePlanner().plan(scenario)

    assert scenario.preferences.bequest_mode is BequestMode.FIXED
    assert scenario.preferences.fixed_bequest == 1_397_805.8599418863
    assert plan.human_capital == pytest.approx(2_725_962.802438837, abs=1.0)
    assert plan.consumption_liability == pytest.approx(1_240_088.3263878513, abs=1.0)
    assert plan.life_insurance_liability == pytest.approx(483_457.4056891775, abs=1.0)
    assert plan.net_worth == pytest.approx(2_202_417.070361808, abs=2.0)
    assert plan.consumption_divisor == pytest.approx(26.26822588569357, abs=3e-7)
    assert plan.initial_discretionary_consumption == pytest.approx(83_843.388584583, abs=1.0)
    assert plan.unconstrained_allocation.equity == pytest.approx(0.7308615171648961, abs=2e-6)
    assert plan.constrained_allocation.equity == pytest.approx(0.3937901376136663, abs=2e-6)
    assert plan.constrained_allocation.bonds == 0.0
    assert plan.constrained_allocation.total == pytest.approx(1.0)


def test_workbook_social_insurance_is_a_package_root_export() -> None:
    assert package.WorkbookSocialInsurance is WorkbookSocialInsurance
    assert "WorkbookSocialInsurance" in package.__all__


def test_continuous_bequest_plan_is_internally_consistent() -> None:
    scenario = PlanningScenario()
    plan = LifecyclePlanner().plan(scenario)
    liabilities = plan.consumption_liability + plan.life_insurance_liability

    assert plan.bequest > 0
    assert plan.net_worth == pytest.approx(plan.financial_wealth + plan.human_capital - liabilities)
    assert plan.initial_discretionary_consumption == pytest.approx(
        plan.net_worth / plan.consumption_divisor
    )
    assert len(plan.glide_path) == scenario.person.horizon
    assert all(allocation.total == pytest.approx(1.0) for allocation in plan.glide_path)
