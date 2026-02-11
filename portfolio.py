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
        return self.nw

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
    """Leveraged equity portfolio with borrowing costs.

    Leverage of 2x means for every $1 of equity, $1 is borrowed.
    Annual borrowing cost is (leverage - 1) * BORROW_COST_RATE.
    """

    BORROW_COST_RATE: float = 0.02

    def __init__(self, init_nw: float, leverage: float,
                 real_mkt_return_func: Callable[[], float]) -> None:
        super().__init__(init_nw)
        if leverage < 1:
            raise ValueError(f"Leverage must be >= 1, got {leverage}")
        self.leverage = leverage
        self.real_mkt_return_func = real_mkt_return_func

    def pass_year(self) -> None:
        """Apply leveraged market returns minus borrowing costs and record."""
        raw_return = self.real_mkt_return_func()
        fees = (self.leverage - 1) * self.BORROW_COST_RATE
        self.nw *= ((raw_return - 1) * self.leverage + 1) - fees
        self.nw_history.append(self.nw)
