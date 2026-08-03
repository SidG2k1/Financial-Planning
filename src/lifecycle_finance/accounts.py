"""Tax-aware accounts, lots, events, withdrawals, conversions, and asset location."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, fields
from typing import Protocol

import numpy as np
from scipy.optimize import linprog

from .domain import AccountType
from .taxes import TaxPolicy, TaxResult


@dataclass(frozen=True, slots=True)
class Asset:
    symbol: str
    bucket: str
    expected_real_return: float
    dividend_yield: float = 0.0
    qualified_dividend_fraction: float = 1.0

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("asset symbol cannot be empty")
        if not self.bucket:
            raise ValueError("asset bucket cannot be empty")
        for name in (
            "expected_real_return",
            "dividend_yield",
            "qualified_dividend_fraction",
        ):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.dividend_yield < 0:
            raise ValueError("dividend_yield cannot be negative")
        if not 0 <= self.qualified_dividend_fraction <= 1:
            raise ValueError("qualified_dividend_fraction must be in [0, 1]")


@dataclass(slots=True)
class TaxLot:
    asset: str
    shares: float
    basis: float
    acquired_year: int

    def __post_init__(self) -> None:
        if not self.asset:
            raise ValueError("tax-lot asset cannot be empty")
        for name in ("shares", "basis", "acquired_year"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.shares < 0 or self.basis < 0:
            raise ValueError("shares and basis cannot be negative")

    def market_value(self, price: float) -> float:
        return self.shares * price

    @property
    def basis_per_share(self) -> float:
        return self.basis / self.shares if self.shares else 0.0


@dataclass(slots=True)
class Account:
    name: str
    account_type: AccountType
    lots: list[TaxLot] = field(default_factory=list)
    allowed_assets: frozenset[str] | None = None

    def __post_init__(self) -> None:
        self.account_type = AccountType(self.account_type)
        if not self.name:
            raise ValueError("account name cannot be empty")
        if self.allowed_assets is not None:
            disallowed = {lot.asset for lot in self.lots} - self.allowed_assets
            if disallowed:
                raise ValueError(f"{self.name} contains disallowed assets: {disallowed}")

    def market_value(self, prices: Mapping[str, float]) -> float:
        return sum(lot.market_value(prices[lot.asset]) for lot in self.lots)

    def holdings(self, prices: Mapping[str, float]) -> dict[str, float]:
        result: dict[str, float] = {}
        for lot in self.lots:
            result[lot.asset] = result.get(lot.asset, 0.0) + lot.market_value(prices[lot.asset])
        return result

    def buy(self, asset: str, dollars: float, price: float, year: int) -> None:
        for name, value in (("dollars", dollars), ("price", price), ("year", year)):
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if dollars < 0 or price <= 0:
            raise ValueError("dollars must be nonnegative and price positive")
        if self.allowed_assets is not None and asset not in self.allowed_assets:
            raise ValueError(f"{asset} is not allowed in {self.name}")
        if dollars:
            self.lots.append(TaxLot(asset, dollars / price, dollars, year))

    def sell(
        self,
        asset: str,
        dollars: float,
        price: float,
        *,
        sale_year: int,
        method: str = "hifo",
    ) -> tuple[float, float]:
        """Sell up to ``dollars`` and return ``(proceeds, realized_gain)``."""
        result = self._sell(
            asset,
            dollars,
            price,
            method=method,
            sale_year=sale_year,
        )
        return result.proceeds, result.gain

    def _sell(
        self,
        asset: str,
        dollars: float,
        price: float,
        *,
        sale_year: int,
        method: str = "hifo",
    ) -> _SaleResult:
        for name, value in (("dollars", dollars), ("price", price)):
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not np.isfinite(sale_year):
            raise ValueError("sale_year must be finite")
        if dollars < 0 or price <= 0:
            raise ValueError("dollars must be nonnegative and price positive")
        candidates = [lot for lot in self.lots if lot.asset == asset and lot.shares > 0]
        if method == "hifo":
            candidates.sort(key=lambda lot: lot.basis_per_share, reverse=True)
        elif method == "fifo":
            candidates.sort(key=lambda lot: lot.acquired_year)
        else:
            raise ValueError("method must be 'hifo' or 'fifo'")

        remaining = dollars
        proceeds = 0.0
        short_term_gain = 0.0
        long_term_gain = 0.0
        for lot in candidates:
            if remaining <= 1e-10:
                break
            lot_value = lot.market_value(price)
            sold_value = min(remaining, lot_value)
            fraction = sold_value / lot_value if lot_value else 0.0
            sold_shares = lot.shares * fraction
            sold_basis = lot.basis * fraction
            lot.shares -= sold_shares
            lot.basis -= sold_basis
            proceeds += sold_value
            gain = sold_value - sold_basis
            if lot.acquired_year >= sale_year:
                short_term_gain += gain
            else:
                long_term_gain += gain
            remaining -= sold_value
        self.lots = [lot for lot in self.lots if lot.shares > 1e-12]
        return _SaleResult(proceeds, short_term_gain, long_term_gain)

    def liquidate(
        self,
        prices: Mapping[str, float],
        *,
        sale_year: int,
        method: str = "hifo",
    ) -> tuple[float, float]:
        proceeds = 0.0
        gains = 0.0
        for asset, amount in list(self.holdings(prices).items()):
            sold, gain = self.sell(
                asset,
                amount,
                prices[asset],
                method=method,
                sale_year=sale_year,
            )
            proceeds += sold
            gains += gain
        return proceeds, gains


@dataclass(slots=True)
class AccountPortfolio:
    accounts: dict[str, Account]
    prices: dict[str, float]

    def __post_init__(self) -> None:
        for symbol, price in self.prices.items():
            if not symbol:
                raise ValueError("price symbols cannot be empty")
            if not np.isfinite(price) or price <= 0:
                raise ValueError("prices must be finite and positive")

    def clone(self) -> AccountPortfolio:
        return deepcopy(self)

    def market_value(self) -> float:
        return sum(account.market_value(self.prices) for account in self.accounts.values())

    def holdings(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for account in self.accounts.values():
            for asset, value in account.holdings(self.prices).items():
                result[asset] = result.get(asset, 0.0) + value
        return result

    def bucket_weights(self, assets: Mapping[str, Asset]) -> dict[str, float]:
        total = self.market_value()
        result: dict[str, float] = {}
        if total <= 0:
            return result
        for symbol, value in self.holdings().items():
            bucket = assets[symbol].bucket
            result[bucket] = result.get(bucket, 0.0) + value / total
        return result


@dataclass(slots=True)
class TaxLedger:
    wages: float = 0.0
    ordinary_income: float = 0.0
    long_term_capital_gains: float = 0.0
    qualified_dividends: float = 0.0
    interest: float = 0.0
    social_security: float = 0.0
    hsa_penalties: float = 0.0
    short_term_capital_gains: float = 0.0
    nonqualified_dividends: float = 0.0

    def tax(
        self,
        policy: TaxPolicy,
        *,
        short_term_loss_carryforward: float = 0.0,
        long_term_loss_carryforward: float = 0.0,
    ) -> TaxResult:
        if not np.isfinite(self.hsa_penalties) or self.hsa_penalties < 0.0:
            raise ValueError("hsa_penalties must be finite and nonnegative")
        base = policy.calculate(
            wages=self.wages,
            ordinary_income=self.ordinary_income,
            short_term_capital_gains=self.short_term_capital_gains,
            long_term_capital_gains=self.long_term_capital_gains,
            qualified_dividends=self.qualified_dividends,
            nonqualified_dividends=self.nonqualified_dividends,
            interest=self.interest,
            social_security=self.social_security,
            short_term_loss_carryforward=short_term_loss_carryforward,
            long_term_loss_carryforward=long_term_loss_carryforward,
        )
        if self.hsa_penalties == 0:
            return base
        return TaxResult(
            base.federal_ordinary + self.hsa_penalties,
            base.federal_capital_gains,
            base.net_investment_income_tax,
            base.state,
            base.payroll,
            base.taxable_social_security,
            base.capital_loss_deduction,
            base.short_term_loss_carryforward,
            base.long_term_loss_carryforward,
        )


class AccountEvent(Protocol):
    def apply(
        self,
        portfolio: AccountPortfolio,
        year: int,
        ledger: TaxLedger,
    ) -> AccountEventResult | None:
        """Mutate the portfolio and record taxable consequences."""


@dataclass(frozen=True, slots=True)
class Contribution:
    account: str
    asset: str
    amount: float

    def apply(self, portfolio: AccountPortfolio, year: int, ledger: TaxLedger) -> None:
        portfolio.accounts[self.account].buy(
            self.asset,
            self.amount,
            portfolio.prices[self.asset],
            year,
        )


@dataclass(frozen=True, slots=True)
class AccountEventResult:
    requested: float
    executed: float
    shortfall: float


@dataclass(frozen=True, slots=True)
class RothConversion:
    source: str
    destination: str
    amount: float
    cash_asset: str

    def apply(
        self,
        portfolio: AccountPortfolio,
        year: int,
        ledger: TaxLedger,
    ) -> AccountEventResult:
        source = portfolio.accounts[self.source]
        destination = portfolio.accounts[self.destination]
        if source.account_type is not AccountType.TRADITIONAL:
            raise ValueError("Roth conversion source must be traditional")
        if destination.account_type is not AccountType.ROTH:
            raise ValueError("Roth conversion destination must be Roth")
        if not np.isfinite(self.amount):
            raise ValueError("amount must be finite")
        if self.amount < 0:
            raise ValueError("amount cannot be negative")
        cash_price = portfolio.prices[self.cash_asset]
        destination.buy(self.cash_asset, 0.0, cash_price, year)
        sale = _raise_cash(source, self.amount, portfolio.prices, year=year)
        destination.buy(
            self.cash_asset,
            sale.proceeds,
            cash_price,
            year,
        )
        ledger.ordinary_income += sale.proceeds
        return AccountEventResult(
            requested=self.amount,
            executed=sale.proceeds,
            shortfall=self.amount - sale.proceeds,
        )


@dataclass(frozen=True, slots=True)
class Withdrawal:
    account: str
    amount: float
    qualified_hsa: bool = True

    def apply(
        self,
        portfolio: AccountPortfolio,
        year: int,
        ledger: TaxLedger,
    ) -> AccountEventResult:
        account = portfolio.accounts[self.account]
        sale = _raise_cash(account, self.amount, portfolio.prices, year=year)
        if account.account_type is AccountType.TAXABLE:
            ledger.short_term_capital_gains += sale.short_term_gain
            ledger.long_term_capital_gains += sale.long_term_gain
        elif account.account_type is AccountType.TRADITIONAL:
            ledger.ordinary_income += sale.proceeds
        elif account.account_type is AccountType.HSA and not self.qualified_hsa:
            ledger.ordinary_income += sale.proceeds
            ledger.hsa_penalties += 0.20 * sale.proceeds
        return AccountEventResult(
            requested=self.amount,
            executed=sale.proceeds,
            shortfall=self.amount - sale.proceeds,
        )


@dataclass(frozen=True, slots=True)
class _SaleResult:
    proceeds: float
    short_term_gain: float
    long_term_gain: float

    @property
    def gain(self) -> float:
        return self.short_term_gain + self.long_term_gain


def _raise_cash(
    account: Account,
    amount: float,
    prices: Mapping[str, float],
    *,
    year: int,
) -> _SaleResult:
    if not np.isfinite(amount):
        raise ValueError("amount must be finite")
    if amount < 0:
        raise ValueError("amount cannot be negative")
    remaining = min(amount, account.market_value(prices))
    proceeds = 0.0
    short_term_gain = 0.0
    long_term_gain = 0.0
    for asset, value in sorted(account.holdings(prices).items()):
        sale = account._sell(
            asset,
            min(remaining, value),
            prices[asset],
            sale_year=year,
        )
        proceeds += sale.proceeds
        short_term_gain += sale.short_term_gain
        long_term_gain += sale.long_term_gain
        remaining -= sale.proceeds
        if remaining <= 1e-8:
            break
    return _SaleResult(proceeds, short_term_gain, long_term_gain)


def rebalance_asset_location(
    portfolio: AccountPortfolio,
    assets: Mapping[str, Asset],
    target_buckets: Mapping[str, float],
    *,
    year: int,
    account_names: tuple[str, ...] | None = None,
    ledger: TaxLedger | None = None,
) -> dict[tuple[str, str], float]:
    """Minimize bucket deviation, then turnover, subject to menus and budgets."""
    if not np.isfinite(year):
        raise ValueError("year must be finite")
    names = tuple(portfolio.accounts) if account_names is None else account_names
    symbols = tuple(assets)
    buckets = tuple(target_buckets)
    if (
        any(not np.isfinite(value) or value < 0.0 for value in target_buckets.values())
        or not np.isclose(sum(target_buckets.values()), 1.0)
    ):
        raise ValueError("target_buckets must sum to one")
    total = sum(portfolio.accounts[name].market_value(portfolio.prices) for name in names)
    if total <= 0:
        return {}

    feasible = [
        (name, symbol)
        for name in names
        for symbol in symbols
        if (allowed := portfolio.accounts[name].allowed_assets) is None or symbol in allowed
    ]
    n_x = len(feasible)
    n_b = len(buckets)
    objective = np.concatenate([np.zeros(n_x), np.ones(2 * n_b)])

    equalities: list[np.ndarray] = []
    rhs: list[float] = []
    for name in names:
        row = np.zeros(n_x + 2 * n_b)
        for index, (account_name, _) in enumerate(feasible):
            if account_name == name:
                row[index] = 1.0
        equalities.append(row)
        rhs.append(portfolio.accounts[name].market_value(portfolio.prices))

    for bucket_index, bucket in enumerate(buckets):
        row = np.zeros(n_x + 2 * n_b)
        for index, (_, symbol) in enumerate(feasible):
            if assets[symbol].bucket == bucket:
                row[index] = 1.0
        row[n_x + bucket_index] = -1.0
        row[n_x + n_b + bucket_index] = 1.0
        equalities.append(row)
        rhs.append(target_buckets[bucket] * total)

    solution = linprog(
        objective,
        A_eq=np.vstack(equalities),
        b_eq=np.asarray(rhs),
        bounds=(0.0, None),
        method="highs",
    )
    if not solution.success:
        raise RuntimeError(f"asset-location optimization failed: {solution.message}")

    current_holdings = {
        name: portfolio.accounts[name].holdings(portfolio.prices) for name in names
    }
    n_primary = n_x + 2 * n_b
    turnover_objective = np.concatenate([np.zeros(n_primary), np.ones(n_x)])
    turnover_constraints: list[np.ndarray] = []
    turnover_rhs: list[float] = []
    for index, (name, symbol) in enumerate(feasible):
        current = current_holdings[name].get(symbol, 0.0)
        positive = np.zeros(n_primary + n_x)
        positive[index] = 1.0
        positive[n_primary + index] = -1.0
        turnover_constraints.append(positive)
        turnover_rhs.append(current)

        negative = np.zeros(n_primary + n_x)
        negative[index] = -1.0
        negative[n_primary + index] = -1.0
        turnover_constraints.append(negative)
        turnover_rhs.append(-current)

    primary_bound = np.zeros(n_primary + n_x)
    primary_bound[n_x:n_primary] = 1.0
    turnover_constraints.append(primary_bound)
    turnover_rhs.append(float(solution.fun) + 1e-8)
    turnover_equalities = np.hstack(
        [np.vstack(equalities), np.zeros((len(equalities), n_x))]
    )
    turnover_solution = linprog(
        turnover_objective,
        A_ub=np.vstack(turnover_constraints),
        b_ub=np.asarray(turnover_rhs),
        A_eq=turnover_equalities,
        b_eq=np.asarray(rhs),
        bounds=(0.0, None),
        method="highs",
    )
    if not turnover_solution.success:
        raise RuntimeError(
            f"asset-location turnover optimization failed: {turnover_solution.message}"
        )

    tax_location_objective = np.zeros(n_primary + n_x)
    for index, (name, _) in enumerate(feasible):
        if portfolio.accounts[name].account_type is AccountType.TAXABLE:
            tax_location_objective[n_primary + index] = 1.0
    total_turnover_bound = np.zeros(n_primary + n_x)
    total_turnover_bound[n_primary:] = 1.0
    location_solution = linprog(
        tax_location_objective,
        A_ub=np.vstack([*turnover_constraints, total_turnover_bound]),
        b_ub=np.asarray([*turnover_rhs, float(turnover_solution.fun) + 1e-8]),
        A_eq=turnover_equalities,
        b_eq=np.asarray(rhs),
        bounds=(0.0, None),
        method="highs",
    )
    if not location_solution.success:
        raise RuntimeError(
            f"asset-location tax optimization failed: {location_solution.message}"
        )

    allocations = {
        key: float(location_solution.x[index])
        for index, key in enumerate(feasible)
        if location_solution.x[index] > 1e-8
    }

    working = portfolio.clone()
    tax_ledger = TaxLedger() if ledger is None else deepcopy(ledger)
    for name in names:
        account = working.accounts[name]
        account_holdings = account.holdings(working.prices)
        desired = {
            symbol: allocations.get((name, symbol), 0.0)
            for symbol in symbols
        }
        proceeds = 0.0
        for symbol, dollars in account_holdings.items():
            excess = dollars - desired.get(symbol, 0.0)
            if excess <= 1e-8:
                continue
            sale = account._sell(
                symbol,
                excess,
                working.prices[symbol],
                sale_year=year,
            )
            proceeds += sale.proceeds
            if account.account_type is AccountType.TAXABLE:
                tax_ledger.short_term_capital_gains += sale.short_term_gain
                tax_ledger.long_term_capital_gains += sale.long_term_gain

        purchases = 0.0
        updated = account.holdings(working.prices)
        for symbol, dollars in desired.items():
            deficit = dollars - updated.get(symbol, 0.0)
            if deficit <= 1e-8:
                continue
            account.buy(symbol, deficit, working.prices[symbol], year)
            purchases += deficit
        if not np.isclose(purchases, proceeds, atol=1e-5):
            raise RuntimeError("optimizer failed to preserve an account budget")
    for name in names:
        portfolio.accounts[name].lots = working.accounts[name].lots
    if ledger is not None:
        for ledger_field in fields(TaxLedger):
            setattr(ledger, ledger_field.name, getattr(tax_ledger, ledger_field.name))
    return allocations


@dataclass(frozen=True, slots=True)
class ProjectionYear:
    year: int
    market_value: float
    taxes: TaxResult
    bucket_weights: dict[str, float]


class AccountProjection:
    """Event-driven deterministic account projection."""

    def __init__(self, assets: Mapping[str, Asset], tax_policy: TaxPolicy | None = None):
        self.assets = dict(assets)
        self.tax_policy = TaxPolicy.for_2026() if tax_policy is None else tax_policy

    def _grow(self, portfolio: AccountPortfolio, year: int, ledger: TaxLedger) -> None:
        old_prices = portfolio.prices.copy()
        for symbol, asset in self.assets.items():
            old_price = old_prices[symbol]
            dividend_per_share = old_price * asset.dividend_yield
            dividends: list[tuple[Account, float]] = []
            for account in portfolio.accounts.values():
                shares = sum(lot.shares for lot in account.lots if lot.asset == symbol)
                dividend = shares * dividend_per_share
                if dividend > 0:
                    dividends.append((account, dividend))
            price_return = asset.expected_real_return - asset.dividend_yield
            new_price = old_price * (1.0 + price_return)
            if new_price <= 0:
                raise ValueError(f"asset {symbol} produced a nonpositive price")
            portfolio.prices[symbol] = new_price
            for account, dividend in dividends:
                if account.account_type is AccountType.TAXABLE:
                    ledger.qualified_dividends += dividend * asset.qualified_dividend_fraction
                    ledger.nonqualified_dividends += (
                        dividend * (1.0 - asset.qualified_dividend_fraction)
                    )
                account.buy(symbol, dividend, new_price, year)

    def run(
        self,
        initial: AccountPortfolio,
        *,
        start_year: int,
        end_year: int,
        events: Mapping[int, list[AccountEvent]] | None = None,
    ) -> tuple[AccountPortfolio, tuple[ProjectionYear, ...]]:
        if end_year < start_year:
            raise ValueError("end_year must not precede start_year")
        state = initial.clone()
        schedule = {} if events is None else events
        history: list[ProjectionYear] = []
        short_term_loss_carryforward = 0.0
        long_term_loss_carryforward = 0.0
        for year in range(start_year, end_year + 1):
            ledger = TaxLedger()
            self._grow(state, year, ledger)
            for event in schedule.get(year, []):
                event.apply(state, year, ledger)
            tax = ledger.tax(
                self.tax_policy,
                short_term_loss_carryforward=short_term_loss_carryforward,
                long_term_loss_carryforward=long_term_loss_carryforward,
            )
            short_term_loss_carryforward = tax.short_term_loss_carryforward
            long_term_loss_carryforward = tax.long_term_loss_carryforward
            history.append(
                ProjectionYear(
                    year,
                    state.market_value(),
                    tax,
                    state.bucket_weights(self.assets),
                )
            )
        return state, tuple(history)
