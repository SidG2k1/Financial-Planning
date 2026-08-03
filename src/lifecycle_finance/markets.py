"""Capital-market assumptions and workbook equilibrium transformations."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from .domain import EconomicExposure

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

WORKBOOK_ASSET_NAMES = (
    "us_large_equity",
    "us_small_equity",
    "global_developed_equity",
    "emerging_market_equity",
    "us_bonds",
    "inflation_linked_bonds",
    "municipal_bonds",
    "global_bonds",
    "cash",
)

_WORKBOOK_REFERENCE_WEIGHTS = np.array(
    [
        0.17311229703644826,
        0.07419098444419213,
        0.21891677075053934,
        0.05677302146020211,
        0.18658453502895422,
        0.06219484500965141,
        0.0,
        0.2282275462700125,
        0.0,
    ]
)
_WORKBOOK_STANDARD_DEVIATIONS = np.array(
    [
        0.15416269599192498,
        0.17947067508948727,
        0.16710499214012084,
        0.21421082758357118,
        0.037928394475143415,
        0.05813108850172724,
        0.04483845451503271,
        0.08333311905653007,
        0.005504888662837908,
    ]
)
_WORKBOOK_CORRELATIONS = np.array(
    [
        [
            1,
            0.9175900450,
            0.8576963322,
            0.7205953705,
            0.0453878055,
            0.1571140326,
            0.0886591916,
            0.2551164117,
            -0.1361017825,
        ],
        [
            0.9175900450,
            1,
            0.8689748730,
            0.7545681169,
            0.0319390546,
            0.1684689936,
            0.1215566755,
            0.2335965760,
            -0.0944137786,
        ],
        [
            0.8576963322,
            0.8689748730,
            1,
            0.8502926388,
            0.1128512657,
            0.2291160451,
            0.1684002641,
            0.4405692565,
            -0.0802416706,
        ],
        [
            0.7205953705,
            0.7545681169,
            0.8502926388,
            1,
            0.1095470674,
            0.2291953657,
            0.1558523998,
            0.3779160381,
            -0.0709528340,
        ],
        [
            0.0453878055,
            0.0319390546,
            0.1128512657,
            0.1095470674,
            1,
            0.7857249426,
            0.7373788729,
            0.6294513714,
            0.1632010759,
        ],
        [
            0.1571140326,
            0.1684689936,
            0.2291160451,
            0.2291953657,
            0.7857249426,
            1,
            0.5932234727,
            0.6079177122,
            0.0631783921,
        ],
        [
            0.0886591916,
            0.1215566755,
            0.1684002641,
            0.1558523998,
            0.7373788729,
            0.5932234727,
            1,
            0.4590465706,
            0.0656868155,
        ],
        [
            0.2551164117,
            0.2335965760,
            0.4405692565,
            0.3779160381,
            0.6294513714,
            0.6079177122,
            0.4590465706,
            1,
            0.0431558482,
        ],
        [
            -0.1361017825,
            -0.0944137786,
            -0.0802416706,
            -0.0709528340,
            0.1632010759,
            0.0631783921,
            0.0656868155,
            0.0431558482,
            1,
        ],
    ],
    dtype=float,
)


@dataclass(frozen=True, slots=True)
class CapitalMarketAssumptions:
    asset_names: tuple[str, ...]
    reference_weights: FloatArray
    covariance: FloatArray
    is_equity: BoolArray
    is_global: BoolArray
    risk_free_rate: float = 0.025
    _sigma_sdf: float | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not np.isfinite(self.risk_free_rate):
            raise ValueError("risk_free_rate must be finite")
        for name, array in (
            ("reference_weights", self.reference_weights),
            ("covariance", self.covariance),
            ("is_equity", self.is_equity),
            ("is_global", self.is_global),
        ):
            if not np.all(np.isfinite(np.asarray(array, dtype=float))):
                raise ValueError(f"{name} must contain only finite values")
        n = len(self.asset_names)
        weights = np.array(self.reference_weights, dtype=float, copy=True)
        covariance = np.array(self.covariance, dtype=float, copy=True)
        is_equity = np.array(self.is_equity, dtype=bool, copy=True)
        is_global = np.array(self.is_global, dtype=bool, copy=True)
        if weights.shape != (n,):
            raise ValueError("reference_weights must have one value per asset")
        if covariance.shape != (n, n):
            raise ValueError("covariance must be square with one row per asset")
        if is_equity.shape != (n,) or is_global.shape != (n,):
            raise ValueError("asset flags must have one value per asset")
        if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, atol=1e-8):
            raise ValueError("reference_weights must be nonnegative and sum to one")
        if not np.allclose(covariance, covariance.T, atol=1e-12):
            raise ValueError("covariance must be symmetric")
        eigenvalues = np.linalg.eigvalsh(covariance)
        if eigenvalues.min() < -1e-9:
            raise ValueError("covariance must be positive semidefinite")
        for array in (weights, covariance, is_equity, is_global):
            array.setflags(write=False)
        object.__setattr__(self, "reference_weights", weights)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "is_equity", is_equity)
        object.__setattr__(self, "is_global", is_global)

    @classmethod
    def workbook_defaults(cls) -> CapitalMarketAssumptions:
        covariance = (
            _WORKBOOK_CORRELATIONS
            * _WORKBOOK_STANDARD_DEVIATIONS[:, None]
            * _WORKBOOK_STANDARD_DEVIATIONS[None, :]
        )
        return cls(
            WORKBOOK_ASSET_NAMES,
            _WORKBOOK_REFERENCE_WEIGHTS,
            covariance,
            np.array([True, True, True, True, False, False, False, False, False]),
            np.array([False, False, True, True, False, False, False, True, False]),
            0.025,
        )

    @property
    def stock_portfolio(self) -> FloatArray:
        weights = np.where(self.is_equity, self.reference_weights, 0.0)
        return np.asarray(weights / weights.sum(), dtype=np.float64)

    @property
    def global_equity_fraction(self) -> float:
        stocks = self.stock_portfolio
        return float(stocks[self.is_global].sum())

    def asset_mix(self, exposure: EconomicExposure) -> FloatArray:
        equity = self.is_equity
        global_equity = equity & self.is_global
        domestic_equity = equity & ~self.is_global
        non_equity = ~equity
        result = np.zeros(len(self.asset_names), dtype=float)
        groups = (
            (global_equity, exposure.equity_fraction * exposure.global_fraction_of_equity),
            (
                domestic_equity,
                exposure.equity_fraction * (1.0 - exposure.global_fraction_of_equity),
            ),
            (non_equity, 1.0 - exposure.equity_fraction),
        )
        for mask, total in groups:
            reference_total = self.reference_weights[mask].sum()
            if total > 0 and reference_total <= 0:
                raise ValueError("reference portfolio has no weight for a requested asset group")
            if reference_total > 0:
                result[mask] = total * self.reference_weights[mask] / reference_total
        return result

    def variance(self, weights: FloatArray) -> float:
        vector = np.asarray(weights, dtype=float)
        return float(vector @ self.covariance @ vector)

    def covariance_between(self, left: FloatArray, right: FloatArray) -> float:
        return float(np.asarray(left) @ self.covariance @ np.asarray(right))

    @staticmethod
    def _stock_sd_from_sdf(real_rate: float, sigma_sdf: float) -> float:
        omega = np.exp(sigma_sdf**2)
        return float((1.0 + real_rate) * omega * np.sqrt(omega - 1.0))

    def sigma_sdf(self) -> float:
        if self._sigma_sdf is not None:
            return self._sigma_sdf
        stock_sd = np.sqrt(self.variance(self.stock_portfolio))
        sigma_sdf = float(
            brentq(
                lambda sigma: self._stock_sd_from_sdf(self.risk_free_rate, sigma) - stock_sd,
                0.0,
                2.0 * stock_sd,
            )
        )
        object.__setattr__(self, "_sigma_sdf", sigma_sdf)
        return sigma_sdf

    def equilibrium_discount_rate(self, exposure_weights: FloatArray) -> float:
        """Workbook expected return implied by covariance with the stock portfolio."""
        covariance_with_stocks = self.covariance_between(self.stock_portfolio, exposure_weights)
        sigma_sdf = self.sigma_sdf()
        denominator = (1.0 + self.risk_free_rate) ** 2 * np.exp(sigma_sdf**2)
        constant = covariance_with_stocks / denominator
        discriminant = 1.0 + 4.0 * constant
        if discriminant < 0.0:
            raise ValueError(
                "equilibrium_discount_rate has no real solution: exposure_weights is "
                "too negatively covariant with the stock portfolio"
            )
        omega = (1.0 + np.sqrt(discriminant)) / 2.0
        return float((1.0 + self.risk_free_rate) * omega - 1.0)

    def aggregate_covariance(
        self,
        human_capital: EconomicExposure,
        liability: EconomicExposure,
    ) -> FloatArray:
        portfolios = np.vstack(
            [
                self.stock_portfolio,
                self.asset_mix(human_capital),
                self.asset_mix(liability),
            ]
        )
        return portfolios @ self.covariance @ portfolios.T


def price_power_claim(risk_tolerance: float, real_rate: float, sigma_sdf: float) -> float:
    """Price of a security paying the stochastic discount factor to ``-risk_tolerance``."""
    return float(
        (1.0 + real_rate) ** (risk_tolerance - 1.0)
        * np.exp(0.5 * risk_tolerance * (risk_tolerance - 1.0) * sigma_sdf**2)
    )


def certainty_equivalent_return(
    risk_tolerance: float,
    real_rate: float,
    sigma_sdf: float,
) -> float:
    if np.isclose(risk_tolerance, 1.0):
        return float((1.0 + real_rate) * np.exp(0.5 * sigma_sdf**2) - 1.0)
    price = price_power_claim(risk_tolerance, real_rate, sigma_sdf)
    return float(price ** (1.0 / (risk_tolerance - 1.0)) - 1.0)
