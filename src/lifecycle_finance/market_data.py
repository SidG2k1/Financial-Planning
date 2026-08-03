"""Optional current-price adapters; core calculations never download data."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Protocol


class PriceProvider(Protocol):
    def prices(self, symbols: tuple[str, ...]) -> dict[str, float]:
        """Return one positive current price for each requested symbol."""


@dataclass(frozen=True, slots=True)
class StaticPriceProvider:
    values: dict[str, float]

    def prices(self, symbols: tuple[str, ...]) -> dict[str, float]:
        missing = set(symbols) - self.values.keys()
        if missing:
            raise KeyError(f"missing static prices for {sorted(missing)}")
        result = {symbol: float(self.values[symbol]) for symbol in symbols}
        if any(not isfinite(price) or price <= 0 for price in result.values()):
            raise ValueError("prices must be positive")
        return result


@dataclass(frozen=True, slots=True)
class YFinancePriceProvider:
    """Lazy yfinance adapter installed with ``lifecycle-finance[market]``."""

    period: str = "5d"

    def prices(self, symbols: tuple[str, ...]) -> dict[str, float]:
        try:
            import yfinance as yf
        except ImportError as error:
            raise RuntimeError("Install the market extra: uv sync --extra market") from error
        if not symbols:
            return {}
        downloaded = yf.download(
            list(symbols),
            period=self.period,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        close = downloaded["Close"]
        result: dict[str, float] = {}
        if getattr(close, "ndim", 1) == 1:
            values = close.dropna()
            if values.empty:
                raise RuntimeError(f"no price returned for {symbols[0]}")
            result[symbols[0]] = float(values.iloc[-1])
        else:
            for symbol in symbols:
                values = close[symbol].dropna()
                if values.empty:
                    raise RuntimeError(f"no price returned for {symbol}")
                result[symbol] = float(values.iloc[-1])
        if any(not isfinite(price) or price <= 0 for price in result.values()):
            raise ValueError("prices must be positive")
        return result
