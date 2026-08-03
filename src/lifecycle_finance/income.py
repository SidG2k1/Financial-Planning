"""Salary projection and Social Security estimates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .domain import Education, IncomePlan, Person, Sex

FloatArray = NDArray[np.float64]


class IncomeModel(Protocol):
    def project(self, person: Person, income: IncomePlan) -> FloatArray:
        """Return real earned income for each year in the planning horizon."""


class RetirementBenefitModel(Protocol):
    def annual_benefit(
        self,
        person: Person,
        earnings: list[float] | FloatArray,
    ) -> float:
        """Return the annual real benefit at the selected claiming age."""


# Workbook polynomial coefficients, ordered from degree zero through four. The original
# formula applies scales [1, 1, .01, .001, .0001] before exponentiation.
_SALARY_COEFFICIENTS: dict[tuple[Education, Sex], tuple[float, ...]] = {
    (Education.GENERIC, Sex.MALE): (7.27627, 0.236958, -0.575408, 0.065291, -0.003012),
    (Education.GENERIC, Sex.FEMALE): (
        7.396984,
        0.258253,
        -0.759774,
        0.100645,
        -0.005097,
    ),
    (Education.HIGH_SCHOOL, Sex.MALE): (
        8.024823,
        0.182566,
        -0.529951,
        0.076131,
        -0.004392,
    ),
    (Education.HIGH_SCHOOL, Sex.FEMALE): (
        9.842184,
        0.004165,
        0.062806,
        -0.015404,
        0.001003,
    ),
    (Education.COLLEGE, Sex.MALE): (
        6.719267,
        0.29959,
        -0.775848,
        0.092649,
        -0.004368,
    ),
    (Education.COLLEGE, Sex.FEMALE): (
        7.476994,
        0.253529,
        -0.746513,
        0.100216,
        -0.005176,
    ),
    (Education.POST_COLLEGE, Sex.MALE): (
        6.223654,
        0.306952,
        -0.603368,
        0.043638,
        -0.000689,
    ),
    (Education.POST_COLLEGE, Sex.FEMALE): (
        4.309414,
        0.551487,
        -1.722771,
        0.242685,
        -0.012938,
    ),
}
_COEFFICIENT_SCALES = (1.0, 1.0, 0.01, 0.001, 0.0001)


@dataclass(frozen=True, slots=True)
class WorkbookSalaryModel:
    """Morningstar-style polynomial salary curve, scaled to current salary."""

    minimum_backfill_age: int = 18

    def base_salary(self, age: float, education: Education, sex: Sex) -> float:
        coefficients = _SALARY_COEFFICIENTS[(education, sex)]
        log_salary = sum(
            scale * coefficient * age**degree
            for degree, (coefficient, scale) in enumerate(
                zip(coefficients, _COEFFICIENT_SCALES, strict=True)
            )
        )
        return float(np.exp(log_salary))

    def salary_at_age(self, age: int, person: Person, income: IncomePlan) -> float:
        if age >= person.retirement_age:
            return 0.0
        base_now = self.base_salary(person.current_age, person.education, person.sex)
        base_then = self.base_salary(age, person.education, person.sex)
        return income.current_salary * base_then / base_now

    def project(self, person: Person, income: IncomePlan) -> FloatArray:
        if income.explicit_real_income is not None:
            result = np.zeros(person.horizon, dtype=float)
            supplied = np.asarray(income.explicit_real_income, dtype=float)
            count = min(len(supplied), person.horizon)
            result[:count] = supplied[:count]
            if count < person.retirement_year and count:
                result[count : person.retirement_year] = supplied[count - 1]
            return result
        return np.array(
            [
                self.salary_at_age(age, person, income)
                for age in range(person.current_age, person.maximum_age + 1)
            ],
            dtype=float,
        )

    def backfilled_earnings(
        self,
        person: Person,
        income: IncomePlan,
    ) -> list[float]:
        return [
            self.salary_at_age(age, person, income)
            for age in range(self.minimum_backfill_age, person.retirement_age)
        ]


@dataclass(frozen=True, slots=True)
class SocialSecurityPolicy:
    """Real-dollar Social Security estimate.

    The default PIA bend points and taxable maximum are the published 2026 values.
    Earnings are treated as already wage-indexed real dollars; callers needing an
    official estimate should supply an SSA earnings record.
    """

    first_bend_point_monthly: float = 1_286.0
    second_bend_point_monthly: float = 7_749.0
    taxable_maximum: float = 184_500.0
    averaging_years: int = 35

    @staticmethod
    def full_retirement_age(birth_year: int) -> float:
        if birth_year <= 1937:
            return 65.0
        if birth_year <= 1942:
            return 65.0 + (birth_year - 1937) * 2.0 / 12.0
        if birth_year <= 1954:
            return 66.0
        if birth_year <= 1959:
            return 66.0 + (birth_year - 1954) * 2.0 / 12.0
        return 67.0

    @staticmethod
    def claiming_adjustment(claiming_age: float, full_retirement_age: float) -> float:
        months = round((min(claiming_age, 70.0) - full_retirement_age) * 12)
        if months < 0:
            early = -months
            first = min(early, 36)
            remainder = max(early - 36, 0)
            reduction = first * (5.0 / 9.0) / 100.0 + remainder * (5.0 / 12.0) / 100.0
            return max(0.0, 1.0 - reduction)
        return 1.0 + months * (2.0 / 3.0) / 100.0

    def primary_insurance_amount(self, earnings: list[float] | FloatArray) -> float:
        capped = sorted(
            (min(max(float(value), 0.0), self.taxable_maximum) for value in earnings),
            reverse=True,
        )
        annual_top = (capped[: self.averaging_years] + [0.0] * self.averaging_years)[
            : self.averaging_years
        ]
        aime = sum(annual_top) / (self.averaging_years * 12.0)
        first = min(aime, self.first_bend_point_monthly)
        second = min(
            max(aime - self.first_bend_point_monthly, 0.0),
            self.second_bend_point_monthly - self.first_bend_point_monthly,
        )
        third = max(aime - self.second_bend_point_monthly, 0.0)
        return 0.90 * first + 0.32 * second + 0.15 * third

    def annual_benefit(
        self,
        person: Person,
        earnings: list[float] | FloatArray,
    ) -> float:
        monthly_pia = self.primary_insurance_amount(earnings)
        adjustment = self.claiming_adjustment(
            float(person.claiming_age),
            self.full_retirement_age(person.birth_year),
        )
        return 12.0 * monthly_pia * adjustment


@dataclass(frozen=True, slots=True)
class WorkbookSocialInsurance:
    """Compatibility implementation of the workbook's annual bend-point method."""

    annual_bend_points: tuple[float, float, float] = (9_912.0, 59_760.0, 117_000.0)
    payout_rates: tuple[float, float, float] = (0.90, 0.32, 0.15)

    @staticmethod
    def _full_benefit_age(person: Person) -> int:
        if person.birth_year < 1940:
            return 65
        if person.birth_year > 1956:
            return 67
        return 66

    @staticmethod
    def _claiming_multiplier(age_difference: int) -> float:
        lookup = {
            -5: 0.70,
            -4: 0.75,
            -3: 0.80,
            -2: 0.8666666666666667,
            -1: 0.9333333333333333,
            0: 1.0,
            1: 1.08,
            2: 1.16,
        }
        if age_difference < -5:
            return 0.70
        if age_difference > 2:
            return 1.24
        return lookup[age_difference]

    def annual_benefit(
        self,
        person: Person,
        earnings: list[float] | FloatArray,
    ) -> float:
        if not len(earnings):
            return 0.0
        highest = sorted((max(float(value), 0.0) for value in earnings), reverse=True)[:35]
        average = sum(highest) / len(highest)
        b1, b2, b3 = self.annual_bend_points
        bands = (
            min(average, b1),
            min(max(average - b1, 0.0), b2 - b1),
            min(max(average - b2, 0.0), b3 - b2),
        )
        full_benefit = sum(
            value * rate for value, rate in zip(bands, self.payout_rates, strict=True)
        )
        difference = person.retirement_age - self._full_benefit_age(person)
        return float(full_benefit * self._claiming_multiplier(difference))


def combined_income_path(
    person: Person,
    income: IncomePlan,
    salary_model: IncomeModel,
    social_security: RetirementBenefitModel,
) -> tuple[FloatArray, float]:
    earned = salary_model.project(person, income)
    if not income.social_security_enabled:
        return earned, 0.0
    if income.explicit_real_income is not None:
        history = list(earned[: person.retirement_year])
    elif isinstance(salary_model, WorkbookSalaryModel):
        history = salary_model.backfilled_earnings(person, income)
    else:
        history = list(earned[: person.retirement_year])
    benefit = social_security.annual_benefit(person, history)
    result = earned.copy()
    payment_age = (
        person.retirement_age
        if isinstance(social_security, WorkbookSocialInsurance)
        else person.claiming_age
    )
    payment_offset = payment_age - person.current_age
    if payment_offset < person.horizon:
        result[max(payment_offset, 0) :] += benefit
    return result, benefit
