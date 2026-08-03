"""Validated domain objects shared by deterministic and stochastic engines."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from math import isfinite
from typing import Any


class Sex(StrEnum):
    MALE = "male"
    FEMALE = "female"


class Education(StrEnum):
    GENERIC = "generic"
    HIGH_SCHOOL = "high_school"
    COLLEGE = "college"
    POST_COLLEGE = "post_college"


class BequestMode(StrEnum):
    OPTIMAL = "optimal"
    FIXED = "fixed"


class LeverageInstrument(StrEnum):
    GENERIC = "generic"
    FUTURES = "futures"
    BOX_SPREAD = "box_spread"


class AccountType(StrEnum):
    TAXABLE = "taxable"
    TRADITIONAL = "traditional"
    ROTH = "roth"
    HSA = "hsa"
    CASH = "cash"


class UtilityCurveKind(StrEnum):
    LINEAR = "linear"
    ISOELASTIC = "isoelastic"
    SPENDING_FLOOR = "spending_floor"
    TARGET = "target"


class OutcomeType(StrEnum):
    """Typed outcomes that utility add-ons may value."""

    SPENDING = "spending"
    TERMINAL_WEALTH = "terminal_wealth"
    BEQUEST = "bequest"
    RETIREMENT_AGE = "retirement_age"
    SOCIAL_SECURITY_CLAIM_AGE = "social_security_claim_age"
    ANNUITIZATION_FRACTION = "annuitization_fraction"
    INSURED_BEQUEST = "insured_bequest"
    LEVERAGE = "leverage"
    ALLOCATION_EQUITY = "allocation_equity"
    RISK_TOLERANCE = "risk_tolerance"
    TIME_PREFERENCE = "time_preference"
    WEDDING_SPEND = "wedding_spend"
    RETIRED = "retired"
    WORKING = "working"


class UtilityAggregation(StrEnum):
    """How a path-by-year utility component becomes one score per path."""

    DISCOUNTED_MEAN = "discounted_mean"
    DISCOUNTED_SUM = "discounted_sum"
    WORST = "worst"
    LAST = "last"


def _finite(name: str, value: float) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")


def _fraction(name: str, value: float) -> None:
    _finite(name, value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")


def _positive(name: str, value: float) -> None:
    _finite(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _nonnegative(name: str, value: float) -> None:
    _finite(name, value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative, got {value}")


@dataclass(frozen=True, slots=True)
class Person:
    current_age: int = 45
    retirement_age: int = 66
    maximum_age: int = 117
    sex: Sex = Sex.FEMALE
    longevity_adjustment: float = 3.0
    education: Education = Education.POST_COLLEGE
    current_year: int = 2026
    social_security_claim_age: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "sex", Sex(self.sex))
        object.__setattr__(self, "education", Education(self.education))
        for name in (
            "current_age",
            "retirement_age",
            "maximum_age",
            "longevity_adjustment",
            "current_year",
        ):
            _finite(name, getattr(self, name))
        if not 0 <= self.current_age < self.retirement_age <= self.maximum_age:
            raise ValueError("ages must satisfy 0 <= current_age < retirement_age <= maximum_age")
        claim_age = self.social_security_claim_age
        if claim_age is None:
            object.__setattr__(
                self,
                "social_security_claim_age",
                min(max(self.retirement_age, 62), 70),
            )
        else:
            _finite("social_security_claim_age", claim_age)
            if not 62 <= claim_age <= 70:
                raise ValueError("social_security_claim_age must be between 62 and 70")

    @property
    def horizon(self) -> int:
        """Number of modeled annual observations, including year zero."""
        return self.maximum_age - self.current_age + 1

    @property
    def retirement_year(self) -> int:
        return self.retirement_age - self.current_age

    @property
    def birth_year(self) -> int:
        return self.current_year - self.current_age

    @property
    def claiming_age(self) -> int:
        assert self.social_security_claim_age is not None
        return self.social_security_claim_age


@dataclass(frozen=True, slots=True)
class FinancialWealth:
    domestic_equity: float = 200_000.0
    global_equity: float = 250_000.0
    bonds: float = 600_000.0
    cash: float = 150_000.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _nonnegative(name, value)

    @property
    def total(self) -> float:
        return self.domestic_equity + self.global_equity + self.bonds + self.cash

    @property
    def weights(self) -> dict[str, float]:
        if self.total == 0:
            return {name: 0.0 for name in ("domestic_equity", "global_equity", "bonds", "cash")}
        return {
            "domestic_equity": self.domestic_equity / self.total,
            "global_equity": self.global_equity / self.total,
            "bonds": self.bonds / self.total,
            "cash": self.cash / self.total,
        }


@dataclass(frozen=True, slots=True)
class IncomePlan:
    current_salary: float = 150_000.0
    defined_contribution: float = 15_000.0
    employer_match_rate: float = 0.50
    explicit_real_income: tuple[float, ...] | None = None
    social_security_enabled: bool = True
    social_security_taxable_max: float = 184_500.0

    def __post_init__(self) -> None:
        if self.explicit_real_income is not None:
            object.__setattr__(
                self,
                "explicit_real_income",
                tuple(map(float, self.explicit_real_income)),
            )
        _nonnegative("current_salary", self.current_salary)
        _nonnegative("defined_contribution", self.defined_contribution)
        _fraction("employer_match_rate", self.employer_match_rate)
        _nonnegative("social_security_taxable_max", self.social_security_taxable_max)
        if self.explicit_real_income is not None:
            for value in self.explicit_real_income:
                _nonnegative("explicit_real_income values", value)

    @property
    def employer_match(self) -> float:
        return self.defined_contribution * self.employer_match_rate


@dataclass(frozen=True, slots=True)
class EconomicExposure:
    equity_fraction: float
    global_fraction_of_equity: float

    def __post_init__(self) -> None:
        _fraction("equity_fraction", self.equity_fraction)
        _fraction("global_fraction_of_equity", self.global_fraction_of_equity)


@dataclass(frozen=True, slots=True)
class Preferences:
    time_preference: float = 0.02
    consumption_elasticity: float = 0.40
    consumption_reference: float = 40_000.0
    risk_tolerance: float | None = None
    spending_floor: float = 40_000.0
    spending_floor_importance: float = 2.0
    spending_floor_scale: float = 10_000.0
    nondiscretionary_consumption: float = 0.0
    annuitization_fraction: float = 1.0
    bequest_flexibility: float = 0.25
    bequest_strength: float = 0.05
    bequest_mode: BequestMode = BequestMode.OPTIMAL
    fixed_bequest: float = 2_000_000.0
    vitality_peak_age: int = 30
    vitality_half_life: float = 35.0
    vitality_floor: float = 0.30
    retirement_utility_multiplier: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "bequest_mode", BequestMode(self.bequest_mode))
        _nonnegative("time_preference", self.time_preference)
        _positive("consumption_elasticity", self.consumption_elasticity)
        _positive("consumption_reference", self.consumption_reference)
        if self.risk_tolerance is not None:
            _positive("risk_tolerance", self.risk_tolerance)
        _nonnegative("spending_floor", self.spending_floor)
        _nonnegative("spending_floor_importance", self.spending_floor_importance)
        _positive("spending_floor_scale", self.spending_floor_scale)
        _nonnegative("nondiscretionary_consumption", self.nondiscretionary_consumption)
        _fraction("annuitization_fraction", self.annuitization_fraction)
        _fraction("bequest_strength", self.bequest_strength)
        _positive("bequest_flexibility", self.bequest_flexibility)
        _nonnegative("fixed_bequest", self.fixed_bequest)
        _finite("vitality_peak_age", self.vitality_peak_age)
        _fraction("vitality_floor", self.vitality_floor)
        _positive("vitality_half_life", self.vitality_half_life)
        _finite("retirement_utility_multiplier", self.retirement_utility_multiplier)
        if self.retirement_utility_multiplier != 1.0:
            raise ValueError(
                "retirement_utility_multiplier is deprecated and must equal 1; "
                "use RETIRED or WORKING utility add-ons"
            )

    @property
    def effective_risk_tolerance(self) -> float:
        """Allocation risk tolerance, derived from CRRA utility unless overridden."""
        if self.risk_tolerance is None:
            return self.consumption_elasticity
        return self.risk_tolerance


@dataclass(frozen=True, slots=True)
class UtilityAddonConfig:
    """Serializable built-in utility add-on specification."""

    name: str
    outcome: OutcomeType
    curve: UtilityCurveKind
    parameters: Mapping[str, float]
    importance: float = 1.0
    aggregation: UtilityAggregation = UtilityAggregation.DISCOUNTED_MEAN
    minimum_age: int | None = None
    maximum_age: int | None = None
    age_reference: float | None = None
    age_growth: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "outcome", OutcomeType(self.outcome))
        object.__setattr__(self, "curve", UtilityCurveKind(self.curve))
        object.__setattr__(
            self,
            "aggregation",
            UtilityAggregation(self.aggregation),
        )
        object.__setattr__(
            self,
            "parameters",
            {name: float(value) for name, value in self.parameters.items()},
        )
        for name, value in self.parameters.items():
            _finite(f"utility parameter {name}", value)
        if not self.name or not self.outcome:
            raise ValueError("utility add-on name and outcome cannot be empty")
        _nonnegative("importance", self.importance)
        required = {
            UtilityCurveKind.LINEAR: set(),
            UtilityCurveKind.ISOELASTIC: {"reference", "elasticity"},
            UtilityCurveKind.SPENDING_FLOOR: {"threshold", "scale"},
            UtilityCurveKind.TARGET: {"target", "tolerance"},
        }[self.curve]
        missing = required - set(self.parameters)
        if missing:
            raise ValueError(f"{self.curve} utility curve is missing parameters: {sorted(missing)}")
        if self.minimum_age is not None:
            _finite("minimum_age", self.minimum_age)
        if self.maximum_age is not None:
            _finite("maximum_age", self.maximum_age)
        if (
            self.minimum_age is not None
            and self.maximum_age is not None
            and self.minimum_age > self.maximum_age
        ):
            raise ValueError("minimum_age cannot exceed maximum_age")
        if self.age_reference is not None and not isfinite(self.age_reference):
            raise ValueError("age_reference must be finite")
        if not isfinite(self.age_growth):
            raise ValueError("age_growth must be finite")
        if self.age_growth != 0.0 and self.age_reference is None:
            raise ValueError("age_reference is required when age_growth is nonzero")


@dataclass(frozen=True, slots=True)
class PlanningScenario:
    person: Person = field(default_factory=Person)
    wealth: FinancialWealth = field(default_factory=FinancialWealth)
    income: IncomePlan = field(default_factory=IncomePlan)
    preferences: Preferences = field(default_factory=Preferences)
    human_capital_exposure: EconomicExposure = field(
        default_factory=lambda: EconomicExposure(0.20, 0.25)
    )
    liability_exposure: EconomicExposure = field(
        default_factory=lambda: EconomicExposure(0.15, 0.0)
    )
    utility_addons: tuple[UtilityAddonConfig, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "utility_addons", tuple(self.utility_addons))


@dataclass(frozen=True, slots=True)
class Allocation:
    domestic_equity: float
    global_equity: float
    bonds: float
    cash: float

    def __post_init__(self) -> None:
        for name in ("domestic_equity", "global_equity", "bonds", "cash"):
            _finite(name, getattr(self, name))

    @property
    def equity(self) -> float:
        return self.domestic_equity + self.global_equity

    @property
    def total(self) -> float:
        return self.domestic_equity + self.global_equity + self.bonds + self.cash

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LifecyclePlan:
    ages: tuple[int, ...]
    survival_probabilities: tuple[float, ...]
    income_path: tuple[float, ...]
    human_capital_path: tuple[float, ...]
    liability_path: tuple[float, ...]
    discretionary_consumption_path: tuple[float, ...]
    human_capital: float
    consumption_liability: float
    life_insurance_liability: float
    financial_wealth: float
    net_worth: float
    social_security_income: float
    consumption_divisor: float
    initial_discretionary_consumption: float
    bequest: float
    current_allocation: Allocation
    unconstrained_allocation: Allocation
    constrained_allocation: Allocation
    glide_path: tuple[Allocation, ...]
    diagnostics: Mapping[str, float | str | bool]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MarketModelConfig:
    equity_risk_premium: float = 0.025
    equity_volatility: float = 0.10
    stock_tail_degrees: float = 5.0
    volatility_persistence: float = 0.80
    volatility_of_volatility: float = 0.25
    stock_bond_correlation: float = -0.20
    initial_real_rate: float = 0.02
    long_run_real_rate: float = 0.02
    rate_mean_reversion: float = 0.15
    rate_volatility: float = 0.012
    minimum_real_rate: float = -0.02
    margin_spread: float = 0.015

    def __post_init__(self) -> None:
        scalar_fields = (
            "equity_risk_premium",
            "equity_volatility",
            "stock_tail_degrees",
            "volatility_persistence",
            "volatility_of_volatility",
            "stock_bond_correlation",
            "initial_real_rate",
            "long_run_real_rate",
            "rate_mean_reversion",
            "rate_volatility",
            "minimum_real_rate",
            "margin_spread",
        )
        for name in scalar_fields:
            if not isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.equity_volatility <= 0:
            raise ValueError("equity_volatility must be positive")
        if self.stock_tail_degrees <= 2:
            raise ValueError("stock_tail_degrees must exceed 2 for finite variance")
        if not -1 <= self.stock_bond_correlation <= 1:
            raise ValueError("stock_bond_correlation must be between -1 and 1")
        if not 0 <= self.volatility_persistence < 1:
            raise ValueError("volatility_persistence must be in [0, 1)")
        _nonnegative("volatility_of_volatility", self.volatility_of_volatility)
        _nonnegative("rate_mean_reversion", self.rate_mean_reversion)
        _nonnegative("rate_volatility", self.rate_volatility)
        _nonnegative("margin_spread", self.margin_spread)


@dataclass(frozen=True, slots=True)
class SimulationSettings:
    paths: int = 1_000
    seed: int = 42
    leverage: float = 1.0
    leverage_instrument: LeverageInstrument = LeverageInstrument.GENERIC
    maintenance_margin: float = 0.25
    margin_call_leverage: float = 1.0
    margin_call_cooldown_years: int = 2
    stochastic_lifespan: bool = True
    stochastic_income: bool = True
    job_loss_probability: float = 0.03
    job_loss_market_sensitivity: float = 5.0
    job_loss_income_fraction: float = 0.0
    antithetic: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "leverage_instrument", LeverageInstrument(self.leverage_instrument)
        )
        _finite("paths", self.paths)
        _finite("seed", self.seed)
        _finite("margin_call_cooldown_years", self.margin_call_cooldown_years)
        if self.paths <= 0:
            raise ValueError("paths must be positive")
        _finite("leverage", self.leverage)
        if self.leverage < 1:
            raise ValueError("leverage must be at least 1")
        _fraction("maintenance_margin", self.maintenance_margin)
        _nonnegative("margin_call_leverage", self.margin_call_leverage)
        if self.margin_call_cooldown_years < 0:
            raise ValueError("margin_call_cooldown_years cannot be negative")
        _fraction("job_loss_probability", self.job_loss_probability)
        _fraction("job_loss_income_fraction", self.job_loss_income_fraction)
        _nonnegative("job_loss_market_sensitivity", self.job_loss_market_sensitivity)
