"""Market-path model interfaces and validated return-model configuration."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol, Self

import numpy as np
from numpy.typing import NDArray

from .calibrated_regime_defaults import (
    EPISTEMIC_INITIAL_PROBABILITIES,
    EPISTEMIC_REGIME_SCALARS,
    EPISTEMIC_TRANSITIONS,
    EQUITY_RISK_PREMIUM,
    EQUITY_VOLATILITY,
    INITIAL_PROBABILITIES,
    INITIAL_REAL_RATE,
    LONG_RUN_REAL_RATE,
    MINIMUM_REAL_RATE,
    RATE_MEAN_REVERSION,
    RATE_VOLATILITY,
    REGIME_CORRELATIONS,
    REGIME_SCALARS,
    TRANSITION,
)
from .domain import MarketModelConfig

FloatArray = NDArray[np.float64]
Seed = int | np.random.SeedSequence
_MINIMUM_UNLEVERED_RETURN = np.nextafter(-1.0, 0.0)
_DEFAULT_MARKET_CONFIG = MarketModelConfig()
_DEFAULT_REGIME_MARKET_CONFIG = MarketModelConfig(
    equity_risk_premium=EQUITY_RISK_PREMIUM,
    equity_volatility=EQUITY_VOLATILITY,
    initial_real_rate=INITIAL_REAL_RATE,
    long_run_real_rate=LONG_RUN_REAL_RATE,
    rate_mean_reversion=RATE_MEAN_REVERSION,
    rate_volatility=RATE_VOLATILITY,
    minimum_real_rate=MINIMUM_REAL_RATE,
)


@dataclass(frozen=True, slots=True)
class MarketPaths:
    equity_returns: FloatArray
    bond_returns: FloatArray
    cash_returns: FloatArray
    real_rates: FloatArray


class MarketPathModel(Protocol):
    config: MarketModelConfig
    bond_duration: float

    def generate(
        self, *, paths: int, horizon: int, seed: Seed, antithetic: bool = True
    ) -> MarketPaths:
        raise NotImplementedError

    def with_config(self, config: MarketModelConfig) -> MarketPathModel:
        raise NotImplementedError

    def with_config_overrides(self, **changes: float) -> MarketPathModel:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class RegimeParameters:
    equity_risk_premium: float
    bond_term_premium: float
    equity_volatility: float
    bond_residual_volatility: float
    rate_volatility: float
    tail_degrees: float
    correlation: FloatArray
    _root: FloatArray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        scalar_fields = (
            "equity_risk_premium",
            "bond_term_premium",
            "equity_volatility",
            "bond_residual_volatility",
            "rate_volatility",
            "tail_degrees",
        )
        for name in scalar_fields:
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        for name in (
            "equity_volatility",
            "bond_residual_volatility",
            "rate_volatility",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.tail_degrees <= 2:
            raise ValueError("tail_degrees must exceed 2 for finite variance")

        correlation = np.array(self.correlation, dtype=float, copy=True)
        if correlation.shape != (3, 3):
            raise ValueError("correlation must have shape (3, 3)")
        if not np.all(np.isfinite(correlation)) or np.any((correlation < -1) | (correlation > 1)):
            raise ValueError("correlation entries must be finite and in [-1, 1]")
        if not np.allclose(correlation, correlation.T):
            raise ValueError("correlation must be symmetric")
        if not np.allclose(np.diag(correlation), 1.0):
            raise ValueError("correlation must have a unit diagonal")
        try:
            root = np.linalg.cholesky(correlation)
        except np.linalg.LinAlgError:
            eigenvalues, eigenvectors = np.linalg.eigh(correlation)
            eigenvalue_scale = float(np.max(np.abs(eigenvalues)))
            tolerance = (
                np.finfo(eigenvalues.dtype).eps
                * correlation.shape[0]
                * eigenvalue_scale
            )
            if np.min(eigenvalues) < -tolerance:
                raise ValueError(
                    "correlation must be positive semidefinite"
                ) from None
            numerical_nonnegative = np.maximum(eigenvalues, 0.0)
            root = (
                eigenvectors * np.sqrt(numerical_nonnegative)
            ) @ eigenvectors.T
        correlation.setflags(write=False)
        root.setflags(write=False)
        object.__setattr__(self, "correlation", correlation)
        object.__setattr__(self, "_root", root)


@dataclass(frozen=True, slots=True)
class EpistemicRegimeDistribution:
    regime_scalars: FloatArray
    transition: FloatArray
    initial_probabilities: FloatArray
    center_transition: FloatArray | None = None
    center_initial_probabilities: FloatArray | None = None

    def __post_init__(self) -> None:
        scalars = np.array(self.regime_scalars, dtype=float, copy=True)
        transition = np.array(self.transition, dtype=float, copy=True)
        initial_probabilities = np.array(
            self.initial_probabilities,
            dtype=float,
            copy=True,
        )
        center_transition = (
            None
            if self.center_transition is None
            else np.array(self.center_transition, dtype=float, copy=True)
        )
        center_initial_probabilities = (
            None
            if self.center_initial_probabilities is None
            else np.array(
                self.center_initial_probabilities,
                dtype=float,
                copy=True,
            )
        )
        if scalars.ndim != 3 or scalars.shape[1:] != (3, 5):
            raise ValueError("epistemic regime_scalars must have shape (samples, 3, 5)")
        samples = scalars.shape[0]
        if samples == 0:
            raise ValueError("epistemic distribution must contain at least one sample")
        if transition.shape != (samples, 3, 3):
            raise ValueError(
                "epistemic transition must have shape (samples, 3, 3)"
            )
        if initial_probabilities.shape != (samples, 3):
            raise ValueError(
                "epistemic initial_probabilities must have shape (samples, 3)"
            )
        if samples > 1 and (
            center_transition is None or center_initial_probabilities is None
        ):
            raise ValueError(
                "multi-sample epistemic distributions must declare their "
                "center_transition and center_initial_probabilities"
            )
        if center_transition is None:
            center_transition = transition[0].copy()
        if center_initial_probabilities is None:
            center_initial_probabilities = initial_probabilities[0].copy()
        if center_transition.shape != (3, 3):
            raise ValueError("epistemic center_transition must have shape (3, 3)")
        if center_initial_probabilities.shape != (3,):
            raise ValueError(
                "epistemic center_initial_probabilities must have shape (3,)"
            )
        for name, values in (
            ("regime_scalars", scalars),
            ("transition", transition),
            ("initial_probabilities", initial_probabilities),
            ("center_transition", center_transition),
            ("center_initial_probabilities", center_initial_probabilities),
        ):
            if not np.all(np.isfinite(values)):
                raise ValueError(f"epistemic {name} must be finite")
        if np.any(scalars[:, :, 2:] < 0.0):
            raise ValueError("epistemic volatility draws must be nonnegative")
        if np.any((transition < 0.0) | (transition > 1.0)):
            raise ValueError("epistemic transition entries must be in [0, 1]")
        if np.any((initial_probabilities < 0.0) | (initial_probabilities > 1.0)):
            raise ValueError(
                "epistemic initial_probabilities entries must be in [0, 1]"
            )
        if not np.allclose(np.sum(transition, axis=2), 1.0):
            raise ValueError("epistemic transition rows must sum to one")
        if not np.allclose(np.sum(initial_probabilities, axis=1), 1.0):
            raise ValueError(
                "epistemic initial_probabilities rows must sum to one"
            )
        if np.any((center_transition < 0.0) | (center_transition > 1.0)):
            raise ValueError(
                "epistemic center_transition entries must be in [0, 1]"
            )
        if np.any(
            (center_initial_probabilities < 0.0)
            | (center_initial_probabilities > 1.0)
        ):
            raise ValueError(
                "epistemic center_initial_probabilities entries must be in [0, 1]"
            )
        if not np.allclose(np.sum(center_transition, axis=1), 1.0):
            raise ValueError("epistemic center_transition rows must sum to one")
        if not np.isclose(np.sum(center_initial_probabilities), 1.0):
            raise ValueError(
                "epistemic center_initial_probabilities must sum to one"
            )
        for values in (
            scalars,
            transition,
            initial_probabilities,
            center_transition,
            center_initial_probabilities,
        ):
            values.setflags(write=False)
        object.__setattr__(self, "regime_scalars", scalars)
        object.__setattr__(self, "transition", transition)
        object.__setattr__(self, "initial_probabilities", initial_probabilities)
        object.__setattr__(self, "center_transition", center_transition)
        object.__setattr__(
            self,
            "center_initial_probabilities",
            center_initial_probabilities,
        )


@dataclass(frozen=True, slots=True)
class RegimeModelConfig:
    normal: RegimeParameters
    stress: RegimeParameters
    inflation_stress: RegimeParameters
    transition: FloatArray
    initial_probabilities: FloatArray
    epistemic: EpistemicRegimeDistribution | None = None

    def __post_init__(self) -> None:
        transition = np.array(self.transition, dtype=float, copy=True)
        initial_probabilities = np.array(self.initial_probabilities, dtype=float, copy=True)
        if transition.shape != (3, 3):
            raise ValueError("transition must have shape (3, 3)")
        if initial_probabilities.shape != (3,):
            raise ValueError("initial_probabilities must have shape (3,)")
        for name, probabilities in (
            ("transition", transition),
            ("initial_probabilities", initial_probabilities),
        ):
            if not np.all(np.isfinite(probabilities)) or np.any(
                (probabilities < 0) | (probabilities > 1)
            ):
                raise ValueError(f"{name} entries must be finite and in [0, 1]")
        if not np.allclose(transition.sum(axis=1), 1.0):
            raise ValueError("transition rows must sum to one")
        if not np.isclose(initial_probabilities.sum(), 1.0):
            raise ValueError("initial_probabilities must sum to one")
        transition.setflags(write=False)
        initial_probabilities.setflags(write=False)
        epistemic = self.epistemic
        if epistemic is None:
            scalars = np.array(
                [
                    [
                        parameters.equity_risk_premium,
                        parameters.bond_term_premium,
                        parameters.equity_volatility,
                        parameters.bond_residual_volatility,
                        parameters.rate_volatility,
                    ]
                    for parameters in self.regimes
                ]
            )
            epistemic = EpistemicRegimeDistribution(
                scalars[np.newaxis, :, :],
                transition[np.newaxis, :, :],
                initial_probabilities[np.newaxis, :],
            )
        else:
            assert epistemic.center_transition is not None
            assert epistemic.center_initial_probabilities is not None
            if not np.allclose(
                transition,
                epistemic.center_transition,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    "transition cannot differ from the retained epistemic center; "
                    "set epistemic=None or supply a matching distribution"
                )
            if not np.allclose(
                initial_probabilities,
                epistemic.center_initial_probabilities,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    "initial_probabilities cannot differ from the retained "
                    "epistemic center; set epistemic=None or supply a matching distribution"
                )
        object.__setattr__(self, "transition", transition)
        object.__setattr__(self, "initial_probabilities", initial_probabilities)
        object.__setattr__(self, "epistemic", epistemic)

    @property
    def regimes(
        self,
    ) -> tuple[RegimeParameters, RegimeParameters, RegimeParameters]:
        return (self.normal, self.stress, self.inflation_stress)

    @classmethod
    def defaults(cls) -> RegimeModelConfig:
        parameters = tuple(
            RegimeParameters(
                *scalars,
                np.array(correlation),
            )
            for scalars, correlation in zip(
                REGIME_SCALARS,
                REGIME_CORRELATIONS,
                strict=True,
            )
        )
        epistemic = EpistemicRegimeDistribution(
            np.array(EPISTEMIC_REGIME_SCALARS),
            np.array(EPISTEMIC_TRANSITIONS),
            np.array(EPISTEMIC_INITIAL_PROBABILITIES),
            np.array(TRANSITION),
            np.array(INITIAL_PROBABILITIES),
        )
        return cls(
            parameters[0],
            parameters[1],
            parameters[2],
            np.array(TRANSITION),
            np.array(INITIAL_PROBABILITIES),
            epistemic,
        )


class StochasticMarket:
    """Fat-tailed equity, stochastic-volatility, mean-reverting-rate model."""

    def __init__(self, config: MarketModelConfig | None = None, bond_duration: float = 6.0):
        if not np.isfinite(bond_duration) or bond_duration <= 0:
            raise ValueError("bond_duration must be positive")
        self.config = _DEFAULT_MARKET_CONFIG if config is None else config
        self.bond_duration = bond_duration

    def with_config(self, config: MarketModelConfig) -> Self:
        return type(self)(config, self.bond_duration)

    def with_config_overrides(self, **changes: float) -> Self:
        """Clone the model while changing only named configuration fields."""
        return self.with_config(replace(self.config, **changes))

    @staticmethod
    def _antithetic_shocks(
        rng: np.random.Generator,
        paths: int,
        horizon: int,
        *,
        distribution: str,
        degrees: float,
        enabled: bool,
    ) -> FloatArray:
        base_paths = (paths + 1) // 2 if enabled else paths
        if distribution == "student":
            shocks = rng.standard_t(degrees, size=(base_paths, horizon))
            shocks *= np.sqrt((degrees - 2.0) / degrees)
        else:
            shocks = rng.standard_normal((base_paths, horizon))
        if enabled:
            shocks = np.concatenate([shocks, -shocks], axis=0)
        return np.asarray(shocks[:paths], dtype=float)

    def generate(
        self,
        *,
        paths: int,
        horizon: int,
        seed: Seed,
        antithetic: bool = True,
    ) -> MarketPaths:
        if paths <= 0:
            raise ValueError("paths must be positive")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        config = self.config
        rng = np.random.default_rng(seed)
        rate_shock = self._antithetic_shocks(
            rng,
            paths,
            horizon,
            distribution="normal",
            degrees=config.stock_tail_degrees,
            enabled=antithetic,
        )
        independent_stock = self._antithetic_shocks(
            rng,
            paths,
            horizon,
            distribution="student",
            degrees=config.stock_tail_degrees,
            enabled=antithetic,
        )
        vol_shock = self._antithetic_shocks(
            rng,
            paths,
            horizon,
            distribution="normal",
            degrees=config.stock_tail_degrees,
            enabled=antithetic,
        )

        correlation = config.stock_bond_correlation
        stock_shock = (
            -correlation * rate_shock
            + np.sqrt(1.0 - correlation**2) * independent_stock
        )
        rates = np.zeros((paths, horizon))
        equity = np.zeros((paths, horizon))
        bonds = np.zeros((paths, horizon))
        cash = np.zeros((paths, horizon))
        current_rate = np.full(
            paths,
            max(config.initial_real_rate, config.minimum_real_rate),
        )
        log_volatility = np.zeros(paths)
        log_volatility_variance = (
            config.volatility_of_volatility**2
            / (1.0 - config.volatility_persistence**2)
        )
        if log_volatility_variance > 0.0:
            log_volatility = self._antithetic_shocks(
                rng,
                paths,
                1,
                distribution="normal",
                degrees=config.stock_tail_degrees,
                enabled=antithetic,
            )[:, 0]
            log_volatility *= np.sqrt(log_volatility_variance)

        for year in range(horizon):
            rates[:, year] = current_rate
            cash[:, year] = current_rate
            log_volatility = (
                config.volatility_persistence * log_volatility
                + config.volatility_of_volatility * vol_shock[:, year]
            )
            volatility = config.equity_volatility * np.exp(
                log_volatility - log_volatility_variance
            )
            equity[:, year] = (
                current_rate + config.equity_risk_premium + volatility * stock_shock[:, year]
            )
            np.maximum(
                equity[:, year],
                _MINIMUM_UNLEVERED_RETURN,
                out=equity[:, year],
            )
            next_rate = (
                current_rate
                + config.rate_mean_reversion * (config.long_run_real_rate - current_rate)
                + config.rate_volatility * rate_shock[:, year]
            )
            next_rate = np.maximum(next_rate, config.minimum_real_rate)
            bonds[:, year] = current_rate - self.bond_duration * (next_rate - current_rate)
            np.maximum(
                bonds[:, year],
                _MINIMUM_UNLEVERED_RETURN,
                out=bonds[:, year],
            )
            current_rate = next_rate
        return MarketPaths(equity, bonds, cash, rates)


class RegimeSwitchingMarket:
    def __init__(
        self,
        config: MarketModelConfig | None = None,
        regime_config: RegimeModelConfig | None = None,
        bond_duration: float = 6.0,
        *,
        epistemic_uncertainty: bool = True,
        epistemic_scenario: int | None = None,
    ) -> None:
        if not np.isfinite(bond_duration) or bond_duration <= 0:
            raise ValueError("bond_duration must be positive")
        self.config = _DEFAULT_REGIME_MARKET_CONFIG if config is None else config
        self.regime_config = (
            RegimeModelConfig.defaults() if regime_config is None else regime_config
        )
        for field_name in (
            "stock_tail_degrees",
            "stock_bond_correlation",
        ):
            if getattr(self.config, field_name) != getattr(
                _DEFAULT_REGIME_MARKET_CONFIG,
                field_name,
            ):
                raise ValueError(
                    f"{field_name} is regime-specific; change RegimeParameters "
                    "instead of MarketModelConfig"
                )
        epistemic_distribution = self.regime_config.epistemic
        assert epistemic_distribution is not None
        if epistemic_scenario is not None:
            if not epistemic_uncertainty:
                raise ValueError(
                    "epistemic_scenario requires epistemic_uncertainty=True"
                )
            if not 0 <= epistemic_scenario < len(
                epistemic_distribution.regime_scalars
            ):
                raise ValueError("epistemic_scenario is out of range")
        self.bond_duration = bond_duration
        self.epistemic_uncertainty = epistemic_uncertainty
        self.epistemic_scenario = epistemic_scenario
        self._parameters = self.regime_config.regimes
        self._roots = np.stack(
            [parameters._root for parameters in self._parameters]
        )
        self._tail_degrees = np.array(
            [parameters.tail_degrees for parameters in self._parameters]
        )

    def with_config(self, config: MarketModelConfig) -> Self:
        return type(self)(
            config,
            self.regime_config,
            self.bond_duration,
            epistemic_uncertainty=self.epistemic_uncertainty,
            epistemic_scenario=self.epistemic_scenario,
        )

    def with_config_overrides(self, **changes: float) -> Self:
        """Clone the model while preserving its calibrated configuration."""
        return self.with_config(replace(self.config, **changes))

    def conditioned_on_epistemic_scenario(self, scenario: int) -> Self:
        """Clone with one fitted parameter scenario shared by every path."""
        return type(self)(
            self.config,
            self.regime_config,
            self.bond_duration,
            epistemic_uncertainty=True,
            epistemic_scenario=scenario,
        )

    def generate(
        self,
        *,
        paths: int,
        horizon: int,
        seed: Seed,
        antithetic: bool = True,
    ) -> MarketPaths:
        if paths <= 0:
            raise ValueError("paths must be positive")
        if horizon <= 0:
            raise ValueError("horizon must be positive")

        config = self.config
        base_paths = (paths + 1) // 2 if antithetic else paths
        rng = np.random.default_rng(seed)
        epistemic_distribution = self.regime_config.epistemic
        assert epistemic_distribution is not None
        epistemic_samples = epistemic_distribution.regime_scalars.shape[0]
        if self.epistemic_uncertainty:
            if self.epistemic_scenario is None:
                base_epistemic = rng.integers(
                    epistemic_samples,
                    size=base_paths,
                    dtype=np.int64,
                )
            else:
                base_epistemic = np.full(
                    base_paths,
                    self.epistemic_scenario,
                    dtype=np.int64,
                )
            initial_draw = rng.random(base_paths)
            initial_probabilities = epistemic_distribution.initial_probabilities[
                base_epistemic
            ]
            base_regime = (
                (initial_draw >= initial_probabilities[:, 0]).astype(np.int64)
                + (
                    initial_draw
                    >= initial_probabilities[:, 0] + initial_probabilities[:, 1]
                )
            )
        else:
            base_epistemic = np.zeros(base_paths, dtype=np.int64)
            base_regime = rng.choice(
                len(self._parameters),
                size=base_paths,
                p=self.regime_config.initial_probabilities,
            )

        mean_scalars = np.array(
            [
                [
                    parameters.equity_risk_premium,
                    parameters.bond_term_premium,
                    parameters.equity_volatility,
                    parameters.bond_residual_volatility,
                    parameters.rate_volatility,
                ]
                for parameters in self._parameters
            ]
        )
        premium_shift = (
            config.equity_risk_premium
            - _DEFAULT_REGIME_MARKET_CONFIG.equity_risk_premium
        )
        equity_volatility_ratio = (
            config.equity_volatility
            / _DEFAULT_REGIME_MARKET_CONFIG.equity_volatility
        )
        rate_volatility_ratio = (
            config.rate_volatility
            / _DEFAULT_REGIME_MARKET_CONFIG.rate_volatility
        )

        rates = np.empty((paths, horizon))
        equity = np.empty((paths, horizon))
        bonds = np.empty((paths, horizon))
        cash = np.empty((paths, horizon))
        current_rate = np.full(
            paths,
            max(config.initial_real_rate, config.minimum_real_rate),
        )
        paired_paths = paths - base_paths
        log_volatility_variance = (
            config.volatility_of_volatility**2
            / (1.0 - config.volatility_persistence**2)
        )
        base_log_volatility = rng.standard_normal(base_paths)
        base_log_volatility *= np.sqrt(log_volatility_variance)
        log_volatility = np.empty(paths)
        log_volatility[:base_paths] = base_log_volatility
        if antithetic:
            log_volatility[base_paths:] = -base_log_volatility[:paired_paths]

        gaussian = np.empty((base_paths, 4))
        normal = gaussian[:, :3]
        volatility_normal = gaussian[:, 3]
        innovations = np.empty((paths, 3))
        base_innovations = innovations[:base_paths]
        student_degrees = np.empty(base_paths)
        student_scale = np.empty(base_paths)
        volatility_shock = np.empty(paths)
        path_parameter = np.empty(paths)
        expected_next_rate = np.empty(paths)
        rate_innovation = np.empty(paths)
        regime = np.empty(paths, dtype=np.int64)
        epistemic = np.empty(paths, dtype=np.int64)
        epistemic[:base_paths] = base_epistemic
        if antithetic:
            epistemic[base_paths:] = base_epistemic[:paired_paths]

        for year in range(horizon):
            rng.standard_normal(out=gaussian)
            np.einsum(
                "nij,nj->ni",
                self._roots[base_regime],
                normal,
                out=base_innovations,
            )
            np.take(self._tail_degrees, base_regime, out=student_degrees)
            student_scale[:] = rng.chisquare(student_degrees)
            student_degrees -= 2.0
            np.divide(student_degrees, student_scale, out=student_scale)
            np.sqrt(student_scale, out=student_scale)
            base_innovations *= student_scale[:, np.newaxis]

            if antithetic:
                innovations[base_paths:] = -base_innovations[:paired_paths]
                volatility_shock[:base_paths] = volatility_normal
                volatility_shock[base_paths:] = -volatility_normal[:paired_paths]
                regime[:base_paths] = base_regime
                regime[base_paths:] = base_regime[:paired_paths]
            else:
                volatility_shock[:] = volatility_normal
                regime[:] = base_regime

            rates[:, year] = current_rate
            cash[:, year] = current_rate
            log_volatility = (
                config.volatility_persistence * log_volatility
                + config.volatility_of_volatility * volatility_shock
            )

            np.subtract(
                log_volatility,
                log_volatility_variance,
                out=volatility_shock,
            )
            np.exp(volatility_shock, out=volatility_shock)
            volatility_shock *= innovations[:, 0]
            if self.epistemic_uncertainty:
                path_parameter[:] = epistemic_distribution.regime_scalars[
                    epistemic,
                    regime,
                    2,
                ]
            else:
                np.take(mean_scalars[:, 2], regime, out=path_parameter)
            path_parameter *= equity_volatility_ratio
            volatility_shock *= path_parameter

            if self.epistemic_uncertainty:
                path_parameter[:] = epistemic_distribution.regime_scalars[
                    epistemic,
                    regime,
                    0,
                ]
            else:
                np.take(mean_scalars[:, 0], regime, out=path_parameter)
            path_parameter += premium_shift
            equity[:, year] = 1.0 + current_rate + path_parameter
            if np.any(equity[:, year] <= 0.0):
                raise ValueError("equity gross-return location must be positive")
            # This is exp(log(location) + log1p(tanh(latent))) - 1, evaluated
            # algebraically to avoid a redundant log/exp round trip.
            np.tanh(volatility_shock, out=volatility_shock)
            volatility_shock += 1.0
            equity[:, year] *= volatility_shock
            equity[:, year] -= 1.0
            np.maximum(
                equity[:, year],
                _MINIMUM_UNLEVERED_RETURN,
                out=equity[:, year],
            )

            if self.epistemic_uncertainty:
                path_parameter[:] = epistemic_distribution.regime_scalars[
                    epistemic,
                    regime,
                    4,
                ]
            else:
                np.take(mean_scalars[:, 4], regime, out=path_parameter)
            path_parameter *= rate_volatility_ratio
            np.multiply(path_parameter, innovations[:, 2], out=rate_innovation)
            expected_next_rate[:] = (
                current_rate
                + config.rate_mean_reversion
                * (config.long_run_real_rate - current_rate)
            )
            next_rate = expected_next_rate + rate_innovation
            next_rate = np.maximum(next_rate, config.minimum_real_rate)
            np.maximum(
                expected_next_rate,
                config.minimum_real_rate,
                out=expected_next_rate,
            )
            np.subtract(next_rate, expected_next_rate, out=rate_innovation)

            if self.epistemic_uncertainty:
                path_parameter[:] = epistemic_distribution.regime_scalars[
                    epistemic,
                    regime,
                    1,
                ]
            else:
                np.take(mean_scalars[:, 1], regime, out=path_parameter)
            bonds[:, year] = (
                current_rate
                + path_parameter
                - self.bond_duration * rate_innovation
            )
            if self.epistemic_uncertainty:
                path_parameter[:] = epistemic_distribution.regime_scalars[
                    epistemic,
                    regime,
                    3,
                ]
            else:
                np.take(mean_scalars[:, 3], regime, out=path_parameter)
            path_parameter *= innovations[:, 1]
            bonds[:, year] += path_parameter
            np.maximum(
                bonds[:, year],
                _MINIMUM_UNLEVERED_RETURN,
                out=bonds[:, year],
            )
            current_rate = next_rate

            if year + 1 < horizon:
                transition_draw = rng.random(base_paths)
                next_regime = np.zeros(base_paths, dtype=np.int64)
                cumulative = np.zeros(base_paths)
                for boundary in range(len(self._parameters) - 1):
                    if self.epistemic_uncertainty:
                        cumulative += epistemic_distribution.transition[
                            base_epistemic,
                            base_regime,
                            boundary,
                        ]
                    else:
                        cumulative += self.regime_config.transition[
                            base_regime,
                            boundary,
                        ]
                    next_regime += transition_draw >= cumulative
                base_regime = next_regime

        return MarketPaths(equity, bonds, cash, rates)
