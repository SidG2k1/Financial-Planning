"""Regression coverage for compatibility and adapter boundary behavior."""

from __future__ import annotations

import json
import sys
import types
from dataclasses import replace

import numpy as np
import pytest

from lifecycle_finance import (
    BequestMode,
    FinancialWealth,
    IncomePlan,
    LifecyclePlanner,
    MonteCarloEngine,
    Person,
    PlanningScenario,
    Preferences,
    SimulationSettings,
    SocialSecurityPolicy,
    YFinancePriceProvider,
)
from lifecycle_finance.cli import main
from lifecycle_finance.income import (
    WorkbookSalaryModel,
    WorkbookSocialInsurance,
    combined_income_path,
)
from lifecycle_finance.sweeps import parameter_sweep, parameter_sweep_2d


def test_workbook_income_uses_retirement_age_while_modern_policy_uses_claiming_age() -> None:
    person = Person(
        current_age=60,
        retirement_age=65,
        maximum_age=70,
        social_security_claim_age=67,
    )
    income = IncomePlan(current_salary=100_000.0)
    salary = WorkbookSalaryModel()

    workbook_path, workbook_benefit = combined_income_path(
        person, income, salary, WorkbookSocialInsurance()
    )
    modern_path, modern_benefit = combined_income_path(
        person, income, salary, SocialSecurityPolicy()
    )

    assert workbook_path[person.retirement_year] == pytest.approx(workbook_benefit)
    assert modern_path[person.retirement_year] == 0.0
    assert modern_path[person.claiming_age - person.current_age] == pytest.approx(modern_benefit)


def test_workbook_benefit_multiplier_uses_retirement_age() -> None:
    person = Person(
        current_age=60,
        retirement_age=65,
        maximum_age=70,
        social_security_claim_age=67,
    )

    benefit = WorkbookSocialInsurance().annual_benefit(person, [10_000.0] * 35)

    assert benefit == pytest.approx(8_948.96 * (13.0 / 15.0))


def test_after_tax_income_preserves_each_social_security_models_payment_start() -> None:
    person = Person(
        current_age=60,
        retirement_age=65,
        maximum_age=70,
        social_security_claim_age=67,
    )
    scenario = PlanningScenario(person=person)
    workbook_engine = MonteCarloEngine()
    workbook_plan = workbook_engine.planner.plan(scenario)
    workbook_income = workbook_engine._after_tax_income(scenario, workbook_plan)
    retirement_year = person.retirement_year
    workbook_taxes = workbook_engine.tax_policy.calculate(
        social_security=workbook_plan.social_security_income,
        include_payroll=False,
    )

    assert workbook_income[retirement_year] == pytest.approx(
        workbook_plan.social_security_income - workbook_taxes.total
    )

    modern_engine = MonteCarloEngine(
        planner=LifecyclePlanner(social_security=SocialSecurityPolicy())
    )
    modern_plan = modern_engine.planner.plan(scenario)
    modern_income = modern_engine._after_tax_income(scenario, modern_plan)

    assert modern_income[retirement_year] == 0.0
    assert modern_income[person.claiming_age - person.current_age] > 0.0


def test_fixed_bequest_is_not_negative_when_liabilities_exceed_wealth() -> None:
    base = PlanningScenario()
    scenario = replace(
        base,
        wealth=FinancialWealth(0.0, 0.0, 0.0, 0.0),
        income=IncomePlan(current_salary=0.0, defined_contribution=0.0),
        preferences=Preferences(
            bequest_mode=BequestMode.FIXED,
            fixed_bequest=100_000.0,
            nondiscretionary_consumption=1_000_000.0,
        ),
    )

    plan = LifecyclePlanner().plan(scenario)

    assert plan.bequest == 0.0


def test_parameter_sweeps_accept_numpy_axes_and_reject_empty_axes() -> None:
    engine = MonteCarloEngine()
    scenario = PlanningScenario()
    settings = SimulationSettings(paths=1, seed=7)

    one_dimensional = parameter_sweep(
        engine,
        scenario,
        settings,
        parameter="leverage",
        values=np.array([1.0, 1.1]),
    )
    two_dimensional = parameter_sweep_2d(
        engine,
        scenario,
        settings,
        x_parameter="leverage",
        x_values=np.array([1.0, 1.1]),
        y_parameter="time_preference",
        y_values=np.array([0.02]),
    )

    assert one_dimensional.values == (1.0, 1.1)
    assert two_dimensional.metrics.shape == (1, 2)
    with pytest.raises(ValueError, match="values cannot be empty"):
        parameter_sweep(
            engine,
            scenario,
            settings,
            parameter="leverage",
            values=np.array([]),
        )
    with pytest.raises(ValueError, match="sweep axes cannot be empty"):
        parameter_sweep_2d(
            engine,
            scenario,
            settings,
            x_parameter="leverage",
            x_values=np.array([]),
            y_parameter="time_preference",
            y_values=np.array([0.02]),
        )


def test_cli_chunk_size_zero_reaches_chunk_validation() -> None:
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        main(["simulate", "--paths", "1", "--chunk-size", "0"])


def test_cli_summary_only_defaults_chunk_size_when_omitted(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["simulate", "--paths", "1", "--summary-only"]) == 0
    assert json.loads(capsys.readouterr().out)["paths"] == 1


class _FakeIndex:
    def __init__(self, value: float) -> None:
        self._value = value

    def __getitem__(self, index: int) -> float:
        assert index == -1
        return self._value


class _FakeClose:
    ndim = 1
    empty = False

    def __init__(self, value: float) -> None:
        self.iloc = _FakeIndex(value)

    def dropna(self) -> _FakeClose:
        return self


@pytest.mark.parametrize("invalid_price", [-1.0, 0.0, float("nan"), float("inf")])
def test_yfinance_price_provider_rejects_nonpositive_or_nonfinite_prices(
    monkeypatch: pytest.MonkeyPatch,
    invalid_price: float,
) -> None:
    fake_yfinance = types.SimpleNamespace(
        download=lambda *args, **kwargs: {"Close": _FakeClose(invalid_price)}
    )
    monkeypatch.setitem(sys.modules, "yfinance", fake_yfinance)

    with pytest.raises(ValueError, match="prices must be positive"):
        YFinancePriceProvider().prices(("TEST",))
