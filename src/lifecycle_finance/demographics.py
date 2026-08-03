"""Mortality, survival, annuity-credit, and insurance calculations."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from .domain import Person, Sex

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class GompertzMortality:
    """Truncated Gompertz mortality calibrated to the lifecycle workbook.

    ``modal_age`` is the mode of age at death and ``dispersion`` controls the
    spread. Survival is conditioned on the modeled person being alive at
    ``current_age`` and is truncated at ``maximum_age``.
    """

    current_age: int
    modal_age: float
    dispersion: float
    maximum_age: int

    def __post_init__(self) -> None:
        for name in ("current_age", "modal_age", "dispersion", "maximum_age"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        for name in ("current_age", "maximum_age"):
            if not isinstance(getattr(self, name), (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
        if self.current_age < 0 or self.maximum_age < self.current_age:
            raise ValueError("ages must satisfy 0 <= current_age <= maximum_age")
        if self.dispersion <= 0.0:
            raise ValueError("dispersion must be positive")

    @classmethod
    def from_person(cls, person: Person, *, adjusted: bool = True) -> GompertzMortality:
        if person.sex is Sex.MALE:
            mode, dispersion = 88.0, 10.6450483499578
        else:
            mode, dispersion = 91.0, 8.8840011008432
        if adjusted:
            mode += person.longevity_adjustment
        return cls(person.current_age, mode, dispersion, person.maximum_age)

    def _raw_survival(self, years: float | FloatArray, attained_age: float) -> float | FloatArray:
        years_array = np.asarray(years, dtype=float)
        exponent = (1.0 - np.exp(years_array / self.dispersion)) * np.exp(
            (attained_age - self.modal_age) / self.dispersion
        )
        result = np.asarray(np.exp(exponent), dtype=np.float64)
        if np.ndim(years) == 0:
            return float(result)
        return result

    def survival(self, from_year: int | float, to_year: int | float) -> float:
        """Probability of survival from one model-year offset to another."""
        if to_year <= from_year:
            return 1.0
        if self.current_age + to_year > self.maximum_age:
            return 0.0
        attained_age = self.current_age + from_year
        terminal = self._raw_survival(
            self.maximum_age + 1.0 - attained_age,
            attained_age,
        )
        untruncated = self._raw_survival(to_year - from_year, attained_age)
        denominator = 1.0 - terminal
        if denominator <= 0:
            return 0.0
        return float(np.clip((untruncated - terminal) / denominator, 0.0, 1.0))

    def survival_curve(self, from_year: int = 0) -> FloatArray:
        final_year = self.maximum_age - self.current_age
        return np.array(
            [self.survival(from_year, year) for year in range(from_year, final_year + 1)],
            dtype=float,
        )

    def one_year_death_probabilities(self) -> FloatArray:
        """Conditional probability of death during each year, including the terminal year."""
        horizon = self.maximum_age - self.current_age
        survival = np.array([self.survival(0, year) for year in range(horizon + 2)])
        return np.asarray(
            np.maximum(survival[:-1] - survival[1:], 0.0),
            dtype=np.float64,
        )

    def sample_death_ages(self, paths: int, rng: np.random.Generator) -> NDArray[np.int64]:
        if paths <= 0:
            raise ValueError("paths must be positive")
        probabilities = self.one_year_death_probabilities()
        probabilities /= probabilities.sum()
        offsets = rng.choice(len(probabilities), size=paths, p=probabilities)
        return np.asarray(self.current_age + offsets + 1, dtype=np.int64)

    def annuity_credit_factor(
        self,
        year: int,
        retirement_year: int,
        annuitization_fraction: float,
    ) -> float:
        """One-year workbook annuity factor applied to post-retirement wealth returns."""
        if year <= retirement_year:
            return 1.0
        q = self.survival(year - 1, year)
        if q <= 0:
            return 0.0
        return 1.0 / (1.0 - annuitization_fraction + annuitization_fraction / q)

    def cumulative_annuity_credit(
        self,
        from_year: int,
        to_year: int,
        retirement_year: int,
        annuitization_fraction: float,
    ) -> float:
        if to_year <= from_year:
            return 1.0
        factor = 1.0
        for year in range(from_year + 1, to_year + 1):
            factor *= self.annuity_credit_factor(year, retirement_year, annuitization_fraction)
        return factor

    def life_annuity_factor(
        self,
        real_rate: float,
        *,
        from_year: int = 0,
        due: bool = True,
    ) -> float:
        """Present value of one survival-contingent real dollar per year."""
        if real_rate <= -1:
            raise ValueError("real_rate must exceed -100%")
        start = from_year if due else from_year + 1
        end = self.maximum_age - self.current_age
        return float(
            sum(
                self.survival(from_year, year) / (1.0 + real_rate) ** (year - from_year)
                for year in range(start, end + 1)
            )
        )

    def term_insurance_price(self, year: int, real_rate: float) -> float:
        self._validate_insurance_inputs(year, real_rate)
        return (1.0 - self.survival(year, year + 1)) / (1.0 + real_rate)

    def permanent_insurance_price(self, year: int, real_rate: float) -> float:
        """Actuarially fair price of one real dollar paid at death."""
        self._validate_insurance_inputs(year, real_rate)
        return _permanent_insurance_price_curve(self, real_rate)[year]

    def permanent_insurance_price_curve(self, real_rate: float) -> FloatArray:
        """Actuarially fair prices of one real dollar paid at death by model year."""
        self._validate_insurance_inputs(0, real_rate)
        return np.array(_permanent_insurance_price_curve(self, real_rate), dtype=np.float64)

    def _validate_insurance_inputs(self, year: int, real_rate: float) -> None:
        final_year = self.maximum_age - self.current_age
        if (
            not isinstance(year, (int, np.integer))
            or isinstance(year, (bool, np.bool_))
            or not 0 <= year <= final_year
        ):
            raise ValueError(f"year must be an integer in [0, {final_year}]")
        if not np.isfinite(real_rate) or real_rate <= -1.0:
            raise ValueError("real_rate must be finite and exceed -100%")


@lru_cache(maxsize=256)
def _permanent_insurance_price_curve(
    mortality: GompertzMortality,
    real_rate: float,
) -> tuple[float, ...]:
    horizon = mortality.maximum_age - mortality.current_age + 1
    discount = 1.0 + real_rate
    result = np.empty(horizon)
    continuation = 0.0
    for year in range(horizon - 1, -1, -1):
        survival = mortality.survival(year, year + 1)
        continuation = ((1.0 - survival) + survival * continuation) / discount
        result[year] = continuation
    return tuple(map(float, result))
