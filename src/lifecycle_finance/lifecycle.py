"""Deterministic lifecycle balance-sheet planner."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

from .allocation import (
    constrain_long_only,
    desired_net_worth_buckets,
    financial_allocation,
)
from .demographics import GompertzMortality
from .domain import (
    Allocation,
    BequestMode,
    LifecyclePlan,
    OutcomeType,
    PlanningScenario,
)
from .income import (
    IncomeModel,
    RetirementBenefitModel,
    WorkbookSalaryModel,
    WorkbookSocialInsurance,
    combined_income_path,
)
from .markets import (
    CapitalMarketAssumptions,
    certainty_equivalent_return,
)
from .utility import (
    UtilityAddon,
    UtilityModel,
    UtilityOutcome,
    consumption_growth_rate,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class _PlanComponents:
    mortality: GompertzMortality
    income: FloatArray
    social_security: float
    human_capital_risky: FloatArray
    human_capital_cash: FloatArray
    liability_risky: FloatArray
    consumption_divisors: FloatArray
    certainty_equivalent_return: float
    consumption_growth: float


class LifecyclePlanner:
    """Workbook-structured lifecycle planner with replaceable assumptions."""

    def __init__(
        self,
        capital_markets: CapitalMarketAssumptions | None = None,
        salary_model: IncomeModel | None = None,
        social_security: RetirementBenefitModel | None = None,
        utility_addons: Sequence[UtilityAddon] = (),
    ) -> None:
        self.capital_markets = (
            CapitalMarketAssumptions.workbook_defaults()
            if capital_markets is None
            else capital_markets
        )
        self.salary_model = WorkbookSalaryModel() if salary_model is None else salary_model
        self.social_security = (
            WorkbookSocialInsurance() if social_security is None else social_security
        )
        self.utility_addons = tuple(utility_addons)

    @staticmethod
    def _present_values(
        values: FloatArray,
        rate: float,
        mortality: GompertzMortality,
        retirement_year: int,
        annuitization_fraction: float,
    ) -> FloatArray:
        horizon = len(values)
        result = np.zeros(horizon)
        result[-1] = values[-1]
        for start in range(horizon - 2, -1, -1):
            one_year_annuity = mortality.annuity_credit_factor(
                start + 1,
                retirement_year,
                annuitization_fraction,
            )
            result[start] = values[start] + one_year_annuity * result[start + 1] / (1.0 + rate)
        return result

    @staticmethod
    def _consumption_divisors(
        mortality: GompertzMortality,
        scenario: PlanningScenario,
        certainty_equivalent: float,
        growth: float,
    ) -> FloatArray:
        person = scenario.person
        preferences = scenario.preferences
        result = np.zeros(person.horizon)
        factor = (1.0 + growth) / (1.0 + certainty_equivalent)
        result[-1] = 1.0
        for start in range(person.horizon - 2, -1, -1):
            survival = mortality.survival(start, start + 1)
            annuity = mortality.annuity_credit_factor(
                start + 1,
                person.retirement_year,
                preferences.annuitization_fraction,
            )
            mortality_weight = survival**preferences.consumption_elasticity * annuity ** (
                1.0 - preferences.consumption_elasticity
            )
            result[start] = 1.0 + mortality_weight * factor * result[start + 1]
        return result

    def _components(self, scenario: PlanningScenario) -> _PlanComponents:
        person = scenario.person
        preferences = scenario.preferences
        mortality = GompertzMortality.from_person(person)
        income_path, social_security = combined_income_path(
            person,
            scenario.income,
            self.salary_model,
            self.social_security,
        )

        human_mix = self.capital_markets.asset_mix(scenario.human_capital_exposure)
        liability_mix = self.capital_markets.asset_mix(scenario.liability_exposure)
        human_rate = self.capital_markets.equilibrium_discount_rate(human_mix)
        liability_rate = self.capital_markets.equilibrium_discount_rate(liability_mix)

        human_capital_risky = self._present_values(
            income_path,
            human_rate,
            mortality,
            person.retirement_year,
            preferences.annuitization_fraction,
        )
        match = np.zeros(person.horizon)
        match[: person.retirement_year] = scenario.income.employer_match
        human_capital_cash = self._present_values(
            match,
            self.capital_markets.risk_free_rate,
            mortality,
            person.retirement_year,
            preferences.annuitization_fraction,
        )
        nondiscretionary = np.full(
            person.horizon,
            preferences.nondiscretionary_consumption,
            dtype=float,
        )
        liability_risky = self._present_values(
            nondiscretionary,
            liability_rate,
            mortality,
            person.retirement_year,
            preferences.annuitization_fraction,
        )

        sigma_sdf = self.capital_markets.sigma_sdf()
        ce_return = certainty_equivalent_return(
            preferences.effective_risk_tolerance,
            self.capital_markets.risk_free_rate,
            sigma_sdf,
        )
        growth = consumption_growth_rate(ce_return, preferences)
        divisors = self._consumption_divisors(
            mortality,
            scenario,
            ce_return,
            growth,
        )
        return _PlanComponents(
            mortality,
            income_path,
            social_security,
            human_capital_risky,
            human_capital_cash,
            liability_risky,
            divisors,
            ce_return,
            growth,
        )

    @staticmethod
    def _discretionary_path(
        initial_consumption: float,
        components: _PlanComponents,
        scenario: PlanningScenario,
    ) -> FloatArray:
        person = scenario.person
        preferences = scenario.preferences
        result = np.zeros(person.horizon)
        result[0] = initial_consumption
        for year in range(1, person.horizon):
            survival = components.mortality.survival(year - 1, year)
            annuity = components.mortality.annuity_credit_factor(
                year,
                person.retirement_year,
                preferences.annuitization_fraction,
            )
            if survival <= 0 or annuity <= 0:
                continue
            one_year_growth = (survival / annuity) ** preferences.consumption_elasticity * (
                1.0 + components.consumption_growth
            )
            result[year] = result[year - 1] * one_year_growth
        return result

    def _choose_bequest(
        self,
        scenario: PlanningScenario,
        components: _PlanComponents,
        wealth_without_insurance: float,
        insurance_price: float,
    ) -> float:
        preferences = scenario.preferences
        maximum = (
            max(wealth_without_insurance / insurance_price, 0.0)
            if insurance_price > 0
            else 0.0
        )
        if preferences.bequest_mode is BequestMode.FIXED:
            return min(preferences.fixed_bequest, maximum)
        if maximum <= 0:
            return 0.0

        survival = components.mortality.survival_curve()
        utility_model = UtilityModel.from_scenario(scenario, self.utility_addons)
        ages = tuple(range(scenario.person.current_age, scenario.person.maximum_age + 1))
        retired = (np.asarray(ages, dtype=float) >= scenario.person.retirement_age)[
            np.newaxis, :
        ].astype(float)
        working = 1.0 - retired

        def objective(bequest: float) -> float:
            remaining = wealth_without_insurance - bequest * insurance_price
            if remaining <= 0:
                return np.inf
            initial = remaining / components.consumption_divisors[0]
            consumption = (
                self._discretionary_path(initial, components, scenario)
                + preferences.nondiscretionary_consumption
            )
            if not np.any(consumption > 0):
                return np.inf
            outcome = UtilityOutcome(
                spending=consumption,
                exposure=survival,
                ages=ages,
                terminal_wealth=np.array([bequest]),
                decisions={
                    OutcomeType.BEQUEST: bequest,
                    OutcomeType.RETIREMENT_AGE: float(scenario.person.retirement_age),
                    OutcomeType.SOCIAL_SECURITY_CLAIM_AGE: float(scenario.person.claiming_age),
                    OutcomeType.ANNUITIZATION_FRACTION: (preferences.annuitization_fraction),
                    OutcomeType.RISK_TOLERANCE: preferences.effective_risk_tolerance,
                    OutcomeType.RETIRED: retired,
                    OutcomeType.WORKING: working,
                },
            )
            return -float(utility_model.score(outcome)[0])

        optimum = minimize_scalar(
            objective,
            bounds=(0.0, maximum * (1.0 - 1e-12)),
            method="bounded",
            options={"xatol": max(1e-6, maximum * 1e-10)},
        )
        if not optimum.success:
            raise RuntimeError(f"bequest optimization failed: {optimum.message}")
        candidates = (0.0, float(optimum.x))
        return min(candidates, key=objective)

    def plan(self, scenario: PlanningScenario | None = None) -> LifecyclePlan:
        if scenario is None:
            scenario = PlanningScenario()
        components = self._components(scenario)
        person = scenario.person
        preferences = scenario.preferences
        wealth = scenario.wealth

        wealth_without_insurance = (
            wealth.total
            + components.human_capital_risky[0]
            + components.human_capital_cash[0]
            - components.liability_risky[0]
        )
        insurance_prices = components.mortality.permanent_insurance_price_curve(
            self.capital_markets.risk_free_rate
        )
        insurance_price = float(insurance_prices[0])
        bequest = self._choose_bequest(
            scenario,
            components,
            wealth_without_insurance,
            insurance_price,
        )
        life_insurance_path = bequest * insurance_prices
        initial_net_worth = wealth_without_insurance - life_insurance_path[0]
        initial_consumption = initial_net_worth / components.consumption_divisors[0]
        discretionary = self._discretionary_path(
            initial_consumption,
            components,
            scenario,
        )
        net_worth_path = components.consumption_divisors * discretionary
        human_total = components.human_capital_risky + components.human_capital_cash
        liability_total = components.liability_risky + life_insurance_path
        financial_path = net_worth_path - human_total + liability_total

        current_weights = wealth.weights
        current = Allocation(**current_weights)
        unconstrained: list[Allocation] = []
        constrained: list[Allocation] = []
        for year in range(person.horizon):
            desired = desired_net_worth_buckets(
                net_worth_path[year],
                preferences.effective_risk_tolerance,
                self.capital_markets.global_equity_fraction,
            )
            allocation = financial_allocation(
                financial_wealth=financial_path[year],
                desired=desired,
                human_capital_risky=components.human_capital_risky[year],
                human_capital_cash=components.human_capital_cash[year],
                human_capital_exposure=scenario.human_capital_exposure,
                liability_risky=components.liability_risky[year],
                liability_cash=life_insurance_path[year],
                liability_exposure=scenario.liability_exposure,
            )
            unconstrained.append(allocation)
            constrained.append(constrain_long_only(allocation))

        human_mix = self.capital_markets.asset_mix(scenario.human_capital_exposure)
        liability_mix = self.capital_markets.asset_mix(scenario.liability_exposure)
        diagnostics: dict[str, float | str | bool] = {
            "human_capital_discount_rate": self.capital_markets.equilibrium_discount_rate(
                human_mix
            ),
            "liability_discount_rate": self.capital_markets.equilibrium_discount_rate(
                liability_mix
            ),
            "certainty_equivalent_return": components.certainty_equivalent_return,
            "discretionary_consumption_growth": components.consumption_growth,
            "insurance_price_per_dollar": insurance_price,
            "bequest_mode": str(preferences.bequest_mode),
            "financial_path_minimum": float(financial_path.min()),
            "nondiscretionary_consumption": preferences.nondiscretionary_consumption,
        }

        ages = tuple(range(person.current_age, person.maximum_age + 1))
        return LifecyclePlan(
            ages=ages,
            survival_probabilities=tuple(map(float, components.mortality.survival_curve())),
            income_path=tuple(map(float, components.income)),
            human_capital_path=tuple(map(float, human_total)),
            liability_path=tuple(map(float, liability_total)),
            discretionary_consumption_path=tuple(map(float, discretionary)),
            human_capital=float(human_total[0]),
            consumption_liability=float(components.liability_risky[0]),
            life_insurance_liability=float(life_insurance_path[0]),
            financial_wealth=wealth.total,
            net_worth=float(initial_net_worth),
            social_security_income=components.social_security,
            consumption_divisor=float(components.consumption_divisors[0]),
            initial_discretionary_consumption=float(initial_consumption),
            bequest=bequest,
            current_allocation=current,
            unconstrained_allocation=unconstrained[0],
            constrained_allocation=constrained[0],
            glide_path=tuple(constrained),
            diagnostics=diagnostics,
        )
