"""Reviewable US federal and state tax estimates for planning scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import inf, isfinite


class FilingStatus(StrEnum):
    SINGLE = "single"
    MARRIED_JOINT = "married_joint"


class _OmittedThreshold(float):
    """Distinguish an omitted threshold without widening the public field type."""


_OMITTED_THRESHOLD = _OmittedThreshold(200_000.0)


@dataclass(frozen=True, slots=True)
class TaxResult:
    federal_ordinary: float
    federal_capital_gains: float
    net_investment_income_tax: float
    state: float
    payroll: float
    taxable_social_security: float
    capital_loss_deduction: float = 0.0
    short_term_loss_carryforward: float = 0.0
    long_term_loss_carryforward: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.federal_ordinary
            + self.federal_capital_gains
            + self.net_investment_income_tax
            + self.state
            + self.payroll
        )


def progressive_tax(amount: float, brackets: tuple[tuple[float, float], ...]) -> float:
    """Tax ``amount`` using inclusive upper bounds and marginal rates."""
    if not isfinite(amount):
        raise ValueError("amount must be finite")
    if not brackets:
        raise ValueError("brackets cannot be empty")
    taxable = max(amount, 0.0)
    lower = 0.0
    result = 0.0
    last_rate = 0.0
    for upper, rate in brackets:
        last_rate = rate
        width = min(taxable, upper) - lower
        if width > 0:
            result += width * rate
        if taxable <= upper:
            break
        lower = upper
    else:
        result += max(taxable - lower, 0.0) * last_rate
    return result


@dataclass(frozen=True, slots=True)
class TaxPolicy:
    filing_status: FilingStatus | str = FilingStatus.SINGLE
    ordinary_brackets: tuple[tuple[float, float], ...] = ()
    capital_gains_brackets: tuple[tuple[float, float], ...] = ()
    standard_deduction: float | None = None
    state_ordinary_rate: float = 0.05
    state_capital_gains_rate: float | None = None
    niit_rate: float = 0.038
    niit_threshold: float = _OMITTED_THRESHOLD
    social_security_wage_base: float = 184_500.0
    social_security_payroll_rate: float = 0.062
    medicare_payroll_rate: float = 0.0145
    additional_medicare_rate: float = 0.009
    additional_medicare_threshold: float = _OMITTED_THRESHOLD

    def __post_init__(self) -> None:
        object.__setattr__(self, "filing_status", FilingStatus(self.filing_status))
        default_threshold = (
            250_000.0
            if self.filing_status is FilingStatus.MARRIED_JOINT
            else 200_000.0
        )
        if self.niit_threshold is _OMITTED_THRESHOLD:
            object.__setattr__(self, "niit_threshold", default_threshold)
        if self.additional_medicare_threshold is _OMITTED_THRESHOLD:
            object.__setattr__(
                self,
                "additional_medicare_threshold",
                default_threshold,
            )
        defaults: TaxPolicy | None = None
        if (
            not self.ordinary_brackets
            or not self.capital_gains_brackets
            or self.standard_deduction is None
        ):
            defaults = self.for_2026(self.filing_status)
            if not self.ordinary_brackets:
                object.__setattr__(self, "ordinary_brackets", defaults.ordinary_brackets)
            if not self.capital_gains_brackets:
                object.__setattr__(self, "capital_gains_brackets", defaults.capital_gains_brackets)
            if self.standard_deduction is None:
                object.__setattr__(self, "standard_deduction", defaults.standard_deduction)
        assert self.standard_deduction is not None
        if not isfinite(self.standard_deduction) or self.standard_deduction < 0:
            raise ValueError("standard_deduction cannot be negative")
        rates = {
            "state_ordinary_rate": self.state_ordinary_rate,
            "niit_rate": self.niit_rate,
            "social_security_payroll_rate": self.social_security_payroll_rate,
            "medicare_payroll_rate": self.medicare_payroll_rate,
            "additional_medicare_rate": self.additional_medicare_rate,
        }
        if self.state_capital_gains_rate is not None:
            rates["state_capital_gains_rate"] = self.state_capital_gains_rate
        for name, rate in rates.items():
            if not isfinite(rate) or not 0.0 <= rate <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        for name, value in {
            "niit_threshold": self.niit_threshold,
            "social_security_wage_base": self.social_security_wage_base,
            "additional_medicare_threshold": self.additional_medicare_threshold,
        }.items():
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name, brackets in (
            ("ordinary_brackets", self.ordinary_brackets),
            ("capital_gains_brackets", self.capital_gains_brackets),
        ):
            previous = 0.0
            for upper, rate in brackets:
                if (
                    (not isfinite(upper) and upper != inf)
                    or upper <= previous
                    or not isfinite(rate)
                    or not 0.0 <= rate <= 1.0
                ):
                    raise ValueError(
                        f"{name} must have increasing positive bounds and finite rates in [0, 1]"
                    )
                previous = upper

    @classmethod
    def for_2026(cls, filing_status: FilingStatus | str = FilingStatus.SINGLE) -> TaxPolicy:
        status = FilingStatus(filing_status)
        rates = (0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37)
        if status is FilingStatus.SINGLE:
            ordinary_limits = (12_400, 50_400, 105_700, 201_775, 256_225, 640_600, inf)
            capital = ((49_450, 0.0), (545_500, 0.15), (inf, 0.20))
            deduction = 16_100.0
            niit_threshold = 200_000.0
            medicare_threshold = 200_000.0
        else:
            ordinary_limits = (24_800, 100_800, 211_400, 403_550, 512_450, 768_700, inf)
            capital = ((98_900, 0.0), (613_700, 0.15), (inf, 0.20))
            deduction = 32_200.0
            niit_threshold = 250_000.0
            medicare_threshold = 250_000.0
        return cls(
            filing_status=status,
            ordinary_brackets=tuple(zip(ordinary_limits, rates, strict=True)),
            capital_gains_brackets=capital,
            standard_deduction=deduction,
            niit_threshold=niit_threshold,
            additional_medicare_threshold=medicare_threshold,
        )

    def taxable_social_security(
        self,
        benefits: float,
        other_ordinary_income: float,
        investment_income: float,
    ) -> float:
        if benefits <= 0:
            return 0.0
        if self.filing_status is FilingStatus.SINGLE:
            first, second, adjustment_cap = 25_000.0, 34_000.0, 4_500.0
        else:
            first, second, adjustment_cap = 32_000.0, 44_000.0, 6_000.0
        provisional = other_ordinary_income + investment_income + 0.5 * benefits
        if provisional <= first:
            return 0.0
        if provisional <= second:
            return min(0.5 * (provisional - first), 0.5 * benefits)
        taxable = 0.85 * (provisional - second) + min(adjustment_cap, 0.5 * benefits)
        return min(0.85 * benefits, taxable)

    def _capital_gains_tax(self, gains: float, taxable_ordinary: float) -> float:
        """Apply preferential brackets with capital gains stacked on ordinary income."""
        gains_remaining = max(gains, 0.0)
        lower = taxable_ordinary
        tax = 0.0
        previous_upper = 0.0
        for upper, rate in self.capital_gains_brackets:
            band_start = max(lower, previous_upper)
            available = max(upper - band_start, 0.0)
            taxed = min(gains_remaining, available)
            tax += taxed * rate
            gains_remaining -= taxed
            if gains_remaining <= 0:
                break
            previous_upper = upper
        if gains_remaining > 0:
            tax += gains_remaining * self.capital_gains_brackets[-1][1]
        return tax

    def calculate(
        self,
        *,
        wages: float = 0.0,
        ordinary_income: float = 0.0,
        short_term_capital_gains: float = 0.0,
        long_term_capital_gains: float = 0.0,
        qualified_dividends: float = 0.0,
        nonqualified_dividends: float = 0.0,
        interest: float = 0.0,
        social_security: float = 0.0,
        short_term_loss_carryforward: float = 0.0,
        long_term_loss_carryforward: float = 0.0,
        include_payroll: bool = True,
    ) -> TaxResult:
        for name, value in {
            "wages": wages,
            "ordinary_income": ordinary_income,
            "qualified_dividends": qualified_dividends,
            "nonqualified_dividends": nonqualified_dividends,
            "interest": interest,
            "social_security": social_security,
            "short_term_loss_carryforward": short_term_loss_carryforward,
            "long_term_loss_carryforward": long_term_loss_carryforward,
        }.items():
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name, value in (
            ("short_term_capital_gains", short_term_capital_gains),
            ("long_term_capital_gains", long_term_capital_gains),
        ):
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")

        short_gain = short_term_capital_gains - short_term_loss_carryforward
        long_gain = long_term_capital_gains - long_term_loss_carryforward
        if short_gain > 0.0 and long_gain < 0.0:
            offset = min(short_gain, -long_gain)
            short_gain -= offset
            long_gain += offset
        elif short_gain < 0.0 and long_gain > 0.0:
            offset = min(-short_gain, long_gain)
            short_gain += offset
            long_gain -= offset

        taxable_short_gain = max(short_gain, 0.0)
        taxable_long_gain = max(long_gain, 0.0)
        preferential = taxable_long_gain + qualified_dividends
        ordinary_before_capital_loss = (
            wages
            + ordinary_income
            + taxable_short_gain
            + nonqualified_dividends
            + interest
        )
        taxable_ss_before_capital_loss = self.taxable_social_security(
            social_security,
            ordinary_before_capital_loss,
            preferential,
        )
        standard_deduction = self.standard_deduction
        assert standard_deduction is not None
        taxable_income_before_capital_loss = max(
            ordinary_before_capital_loss
            + taxable_ss_before_capital_loss
            + preferential
            - standard_deduction,
            0.0,
        )
        capital_loss_deduction = min(
            max(-(short_gain + long_gain), 0.0),
            3_000.0,
            taxable_income_before_capital_loss,
        )
        remaining_deduction = capital_loss_deduction
        if short_gain < 0.0:
            applied = min(-short_gain, remaining_deduction)
            short_gain += applied
            remaining_deduction -= applied
        if long_gain < 0.0 and remaining_deduction > 0.0:
            applied = min(-long_gain, remaining_deduction)
            long_gain += applied

        short_carryforward = max(-short_gain, 0.0)
        long_carryforward = max(-long_gain, 0.0)
        taxable_short_gain = max(short_gain, 0.0)
        taxable_long_gain = max(long_gain, 0.0)
        preferential = taxable_long_gain + qualified_dividends
        other_ordinary = (
            wages
            + ordinary_income
            + taxable_short_gain
            + nonqualified_dividends
            + interest
            - capital_loss_deduction
        )
        taxable_ss = self.taxable_social_security(
            social_security,
            other_ordinary,
            preferential,
        )
        gross_ordinary = other_ordinary + taxable_ss
        ordinary_after_deduction = max(gross_ordinary - standard_deduction, 0.0)
        unused_deduction = max(standard_deduction - gross_ordinary, 0.0)
        taxable_preferential = max(preferential - unused_deduction, 0.0)

        federal_ordinary = progressive_tax(ordinary_after_deduction, self.ordinary_brackets)
        federal_gains = self._capital_gains_tax(taxable_preferential, ordinary_after_deduction)

        modified_agi = gross_ordinary + preferential
        net_investment_income = max(
            taxable_short_gain
            + preferential
            + nonqualified_dividends
            + interest
            - capital_loss_deduction,
            0.0,
        )
        niit = self.niit_rate * min(
            net_investment_income,
            max(modified_agi - self.niit_threshold, 0.0),
        )

        state_gains_rate = (
            self.state_ordinary_rate
            if self.state_capital_gains_rate is None
            else self.state_capital_gains_rate
        )
        state = self.state_ordinary_rate * gross_ordinary + state_gains_rate * preferential

        payroll = 0.0
        if include_payroll:
            payroll = (
                min(wages, self.social_security_wage_base) * self.social_security_payroll_rate
                + wages * self.medicare_payroll_rate
                + max(wages - self.additional_medicare_threshold, 0.0)
                * self.additional_medicare_rate
            )

        return TaxResult(
            federal_ordinary,
            federal_gains,
            niit,
            state,
            payroll,
            taxable_ss,
            capital_loss_deduction,
            short_carryforward,
            long_carryforward,
        )

    def after_tax_earned_income(self, wages: float) -> float:
        return wages - self.calculate(wages=wages).total
