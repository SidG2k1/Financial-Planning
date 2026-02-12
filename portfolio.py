from abc import ABC, abstractmethod
from typing import Callable, List, Tuple


class Portfolio(ABC):
    """Base class for investment portfolios tracking net worth over time.

    All monetary values are in units of $1,000 (2023 USD).
    """

    def __init__(self, init_nw: float) -> None:
        if not isinstance(init_nw, (int, float)) or init_nw != init_nw:
            raise ValueError(f"init_nw must be a finite number, got {init_nw}")
        self.nw: float = float(init_nw)
        self.nw_history: List[float] = [self.nw]

    @abstractmethod
    def pass_year(self) -> None:
        """Simulate one year of portfolio growth/decline and append to history."""
        ...

    def get_nw(self) -> float:
        """Return current net worth in $1,000 units."""
        return float(self.nw)

    def get_nw_history(self) -> List[float]:
        """Return list of net worth values, one per year since inception."""
        return self.nw_history

    def remove_money(self, amount: float) -> None:
        """Withdraw amount ($1,000 units) from the portfolio."""
        if amount < 0:
            raise ValueError(f"Withdrawal amount must be non-negative, got {amount}")
        self.nw -= amount

    def add_money(self, amount: float) -> None:
        """Deposit amount ($1,000 units) into the portfolio."""
        if amount < 0:
            raise ValueError(f"Deposit amount must be non-negative, got {amount}")
        self.nw += amount


class PortfolioManager:
    """Coordinates multiple portfolios, applying the same operations to each."""

    def __init__(self, portfolios: List[Portfolio]) -> None:
        self.portfolios = portfolios

    def pass_year(self) -> None:
        """Advance all portfolios by one year."""
        for portfolio in self.portfolios:
            portfolio.pass_year()

    def get_nw(self) -> List[Tuple[str, float]]:
        """Return (class_name, net_worth) for each portfolio."""
        return [(type(p).__name__, p.get_nw()) for p in self.portfolios]

    def get_nw_history(self) -> List[List[float]]:
        """Return net worth history for each portfolio."""
        return [p.get_nw_history() for p in self.portfolios]

    def remove_money(self, amount: float) -> None:
        """Withdraw the same amount from every portfolio."""
        for portfolio in self.portfolios:
            portfolio.remove_money(amount)

    def add_money(self, amount: float) -> None:
        """Deposit the same amount into every portfolio."""
        for portfolio in self.portfolios:
            portfolio.add_money(amount)


class CashPortfolio(Portfolio):
    """Portfolio holding only cash, losing value to inflation each year."""

    INFLATION_RATE: float = 0.02

    def pass_year(self) -> None:
        """Reduce net worth by inflation rate and record."""
        self.nw /= (1 + self.INFLATION_RATE)
        self.nw_history.append(self.nw)


class StockPortfolio(Portfolio):
    """Portfolio of equities growing at the real market return rate."""

    def __init__(self, init_nw: float, real_mkt_return_func: Callable[[], float]) -> None:
        super().__init__(init_nw)
        self.real_mkt_return_func = real_mkt_return_func

    def pass_year(self) -> None:
        """Apply one year of market returns and record."""
        self.nw *= self.real_mkt_return_func()
        self.nw_history.append(self.nw)


class LeveragedStockPortfolio(Portfolio):
    """Leveraged equity portfolio with dynamic borrowing costs and margin calls.

    Leverage of 2x means for every $1 of equity, $1 is borrowed.
    Annual borrowing cost is (leverage - 1) * margin_fee, where
    margin_fee = bond_yield + broker_spread (set externally each year).

    If maintenance_margin > 0, a margin call is triggered when equity drops
    below the maintenance threshold, forcing partial liquidation and reducing
    leverage for a cooldown period.
    """

    def __init__(self, init_nw: float, leverage: float,
                 real_mkt_return_func: Callable[[], float],
                 margin_fee_func: Callable[[], float],
                 maintenance_margin: float = 0.25,
                 margin_call_leverage: float = 1.0) -> None:
        super().__init__(init_nw)
        if leverage < 1:
            raise ValueError(f"Leverage must be >= 1, got {leverage}")
        self.leverage = leverage
        self.real_mkt_return_func = real_mkt_return_func
        self.margin_fee_func = margin_fee_func
        self.maintenance_margin = maintenance_margin
        self.margin_call_leverage = margin_call_leverage
        self.effective_leverage = leverage
        self.margin_call_count = 0
        self._margin_call_cooldown = 0

    def pass_year(self) -> None:
        """Apply leveraged market returns minus borrowing costs and record."""
        raw_return = self.real_mkt_return_func()
        margin_fee = self.margin_fee_func()
        lev = self.effective_leverage
        fees = (lev - 1) * margin_fee
        new_nw = self.nw * ((raw_return - 1) * lev + 1 - fees)

        # Margin call: check equity-to-gross-assets ratio.
        # debt = starting_equity * (leverage - 1), unchanged by market move.
        # equity_ratio = new_equity / (new_equity + debt).
        margin_called = False
        if self.maintenance_margin > 0 and self.nw > 0 and lev > 1:
            debt = self.nw * (lev - 1)
            gross_assets = new_nw + debt
            equity_ratio = new_nw / gross_assets if gross_assets > 0 else 0.0
            if equity_ratio < self.maintenance_margin:
                # Broker liquidates to de-leverage. Equity stays at actual
                # post-loss level; floor at 0 for limited liability.
                new_nw = max(new_nw, 0.0)
                self.effective_leverage = self.margin_call_leverage
                self.margin_call_count += 1
                self._margin_call_cooldown = 2
                margin_called = True

        if not margin_called and self._margin_call_cooldown > 0:
            self._margin_call_cooldown -= 1
            if self._margin_call_cooldown == 0:
                self.effective_leverage = self.leverage  # restore

        self.nw = new_nw
        self.nw_history.append(self.nw)
