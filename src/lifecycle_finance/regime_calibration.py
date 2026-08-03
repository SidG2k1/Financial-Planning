"""Offline calibration for the three-state market return model."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import pairwise
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

REGIME_NAMES: Final = ("normal", "growth_stress", "inflation_stress")


@dataclass(frozen=True, slots=True)
class HistoricalReturns:
    years: IntArray
    real_equity_returns: FloatArray
    real_cash_returns: FloatArray
    real_bond_returns: FloatArray

    def __post_init__(self) -> None:
        arrays = (
            self.years,
            self.real_equity_returns,
            self.real_cash_returns,
            self.real_bond_returns,
        )
        if any(values.ndim != 1 for values in arrays):
            raise ValueError("historical return inputs must be one-dimensional")
        if len({len(values) for values in arrays}) != 1:
            raise ValueError("historical return inputs must have equal lengths")
        if len(self.years) < 12:
            raise ValueError("at least 12 annual observations are required")
        if not np.all(np.diff(self.years) == 1):
            raise ValueError("years must be consecutive and strictly increasing")
        for values in arrays[1:]:
            if not np.all(np.isfinite(values)):
                raise ValueError("historical returns must be finite")
            if np.any(values <= -1.0):
                raise ValueError("historical simple returns must exceed -100%")


@dataclass(frozen=True, slots=True)
class FittedRegime:
    equity_risk_premium: float
    bond_term_premium: float
    equity_volatility: float
    bond_residual_volatility: float
    rate_volatility: float
    tail_degrees: float
    correlation: FloatArray


@dataclass(frozen=True, slots=True)
class EpistemicCalibration:
    regime_scalars: FloatArray
    transition: FloatArray
    initial_probabilities: FloatArray
    seed: int


@dataclass(frozen=True, slots=True)
class FittedRegimeModel:
    regimes: tuple[FittedRegime, FittedRegime, FittedRegime]
    transition: FloatArray
    initial_probabilities: FloatArray
    labels: IntArray
    stress_log_return_threshold: float
    equity_risk_premium: float
    equity_volatility: float
    initial_real_rate: float
    long_run_real_rate: float
    rate_mean_reversion: float
    rate_volatility: float
    minimum_real_rate: float
    epistemic: EpistemicCalibration
    stress_quantile: float = 0.25
    bond_duration: float = 6.0


def _standardized_t_negative_log_likelihood(
    degrees: float,
    standardized: FloatArray,
) -> float:
    scale = np.sqrt((degrees - 2.0) / degrees)
    values = standardized / scale
    log_density = (
        gammaln((degrees + 1.0) / 2.0)
        - gammaln(degrees / 2.0)
        - 0.5 * np.log(degrees * np.pi)
        - np.log(scale)
        - 0.5 * (degrees + 1.0) * np.log1p(values * values / degrees)
    )
    return float(-np.sum(log_density))


def _fit_tail_degrees(values: FloatArray) -> float:
    standardized = (values - np.mean(values)) / np.std(values, ddof=1)
    fitted = minimize_scalar(
        _standardized_t_negative_log_likelihood,
        args=(standardized,),
        bounds=(4.1, 30.0),
        method="bounded",
        options={"xatol": 1e-8},
    )
    if not fitted.success:
        raise RuntimeError("Student-t tail fit did not converge")
    return float(fitted.x)


def _nearest_correlation(sample: FloatArray, observations: int) -> FloatArray:
    shrinkage = min(0.25, 3.0 / observations)
    shrunk = np.array(sample, copy=True)
    shrunk[0, 1:] *= 1.0 - shrinkage
    shrunk[1:, 0] *= 1.0 - shrinkage
    eigenvalues, eigenvectors = np.linalg.eigh(shrunk)
    positive = (eigenvectors * np.maximum(eigenvalues, 1e-8)) @ eigenvectors.T
    scale = np.sqrt(np.diag(positive))
    correlation = positive / np.outer(scale, scale)
    np.fill_diagonal(correlation, 1.0)
    return np.asarray(correlation, dtype=float)


def _validate_stationary_vector(probabilities: FloatArray) -> FloatArray:
    if np.sum(probabilities) < 0:
        probabilities = -probabilities
    probabilities = probabilities / np.sum(probabilities)
    if np.any(probabilities < -1e-9):
        raise ValueError(
            "transition matrix has no nonnegative stationary distribution; it may be "
            "reducible or periodic. Increase transition_prior to guarantee full support."
        )
    probabilities = np.clip(probabilities, 0.0, None)
    return np.asarray(probabilities / np.sum(probabilities), dtype=float)


def _stationary_probabilities(transition: FloatArray) -> FloatArray:
    eigenvalues, eigenvectors = np.linalg.eig(transition.T)
    index = int(np.argmin(np.abs(eigenvalues - 1.0)))
    probabilities = np.real(eigenvectors[:, index])
    return _validate_stationary_vector(probabilities)


def _bootstrap_epistemic_calibration(
    *,
    labels: IntArray,
    equity_excess: FloatArray,
    equity_log_excess: FloatArray,
    bond_excess: FloatArray,
    rate_surprise: FloatArray,
    bond_duration: float,
    transition_counts: FloatArray,
    target_scalars: FloatArray,
    target_initial_probabilities: FloatArray,
    samples: int,
    seed: int,
) -> EpistemicCalibration:
    if samples <= 0:
        raise ValueError("epistemic_samples must be positive")
    rng = np.random.default_rng(seed)
    scalar_draws = np.empty((samples, 3, 5))
    transition_draws = np.empty((samples, 3, 3))
    initial_draws = np.empty((samples, 3))
    active_indices = [np.flatnonzero(labels == regime_index) for regime_index in range(3)]

    for sample_index in range(samples):
        for regime_index, candidates in enumerate(active_indices):
            selected = rng.choice(candidates, size=len(candidates), replace=True)
            selected_bond_residual = (
                bond_excess[selected] + bond_duration * rate_surprise[selected]
            )
            scalar_draws[sample_index, regime_index] = (
                np.mean(equity_excess[selected]),
                np.mean(bond_excess[selected]),
                max(np.std(equity_log_excess[selected], ddof=1), 1e-6),
                max(np.std(selected_bond_residual, ddof=1), 1e-6),
                max(np.std(rate_surprise[selected], ddof=1), 1e-6),
            )
        for regime_index in range(3):
            transition_draws[sample_index, regime_index] = rng.dirichlet(
                transition_counts[regime_index] + 0.5
            )
        initial_draws[sample_index] = _stationary_probabilities(
            transition_draws[sample_index]
        )

    for column in (0, 1):
        scalar_draws[:, :, column] += (
            target_scalars[:, column]
            - np.mean(scalar_draws[:, :, column], axis=0)
        )
    for column in (2, 3, 4):
        scalar_draws[:, :, column] *= (
            target_scalars[:, column]
            / np.mean(scalar_draws[:, :, column], axis=0)
        )
    for column in (0, 1):
        target_unconditional = float(
            target_initial_probabilities @ target_scalars[:, column]
        )
        sampled_unconditional = float(
            np.mean(
                np.sum(
                    initial_draws * scalar_draws[:, :, column],
                    axis=1,
                )
            )
        )
        scalar_draws[:, :, column] += (
            target_unconditional - sampled_unconditional
        )

    return EpistemicCalibration(
        regime_scalars=scalar_draws,
        transition=transition_draws,
        initial_probabilities=initial_draws,
        seed=seed,
    )


def fit_regime_model(
    history: HistoricalReturns,
    *,
    bond_duration: float = 6.0,
    stress_quantile: float = 0.25,
    transition_prior: float = 0.0,
    rate_floor_quantile: float = 0.01,
    epistemic_samples: int = 128,
    epistemic_seed: int = 20_260_724,
) -> FittedRegimeModel:
    """Fit three annual regimes and the shared real-rate process.

    Stress observations are the bottom ``stress_quantile`` of real equity log
    returns over cash. Positive bond excess returns identify growth stress;
    negative bond excess returns identify inflation stress.
    """

    if not np.isfinite(bond_duration) or bond_duration <= 0:
        raise ValueError("bond_duration must be positive")
    if not 0 < stress_quantile < 0.5:
        raise ValueError("stress_quantile must be between zero and 0.5")
    if not np.isfinite(transition_prior) or transition_prior < 0:
        raise ValueError("transition_prior must be nonnegative")
    if not 0 <= rate_floor_quantile < 0.5:
        raise ValueError("rate_floor_quantile must be in [0, 0.5)")

    equity = history.real_equity_returns[:-1]
    cash = history.real_cash_returns[:-1]
    bonds = history.real_bond_returns[:-1]
    next_cash = history.real_cash_returns[1:]
    equity_log_excess = np.log1p(equity) - np.log1p(cash)
    equity_excess = equity - cash
    bond_excess = bonds - cash
    design = np.column_stack((np.ones(len(cash)), cash))
    intercept, persistence = np.linalg.lstsq(design, next_cash, rcond=None)[0]
    rate_surprise = next_cash - design @ np.array([intercept, persistence])

    threshold = float(np.quantile(equity_log_excess, stress_quantile))
    stress = equity_log_excess <= threshold
    labels = np.zeros(len(equity), dtype=np.int64)
    labels[stress & (bond_excess >= 0.0)] = 1
    labels[stress & (bond_excess < 0.0)] = 2

    counts = np.full((3, 3), transition_prior, dtype=float)
    for current, following in pairwise(labels):
        counts[current, following] += 1.0
    if np.any(np.sum(counts, axis=1) == 0.0):
        raise ValueError("each fitted regime must have an observed outgoing transition")
    transition = counts / np.sum(counts, axis=1, keepdims=True)
    initial_probabilities = _stationary_probabilities(transition)

    regimes: list[FittedRegime] = []
    for regime_index in range(3):
        active = labels == regime_index
        observations = int(np.count_nonzero(active))
        if observations < 4:
            raise ValueError(f"{REGIME_NAMES[regime_index]} has fewer than four observations")
        bond_residual = (
            bond_excess[active] + bond_duration * rate_surprise[active]
        )
        innovations = np.column_stack(
            (
                equity_log_excess[active],
                bond_residual,
                rate_surprise[active],
            )
        )
        correlation = _nearest_correlation(
            np.corrcoef(innovations, rowvar=False),
            observations,
        )
        regimes.append(
            FittedRegime(
                equity_risk_premium=float(np.mean(equity_excess[active])),
                bond_term_premium=float(np.mean(bond_excess[active])),
                equity_volatility=float(np.std(equity_log_excess[active], ddof=1)),
                bond_residual_volatility=float(np.std(bond_residual, ddof=1)),
                rate_volatility=float(np.std(rate_surprise[active], ddof=1)),
                tail_degrees=_fit_tail_degrees(equity_log_excess[active]),
                correlation=correlation,
            )
        )

    mean_reversion = float(np.clip(1.0 - persistence, 0.0, 1.0))
    long_run_rate = (
        float(intercept / mean_reversion)
        if mean_reversion > np.finfo(float).eps
        else float(np.mean(history.real_cash_returns))
    )
    stationary_premiums = np.array(
        [regime.equity_risk_premium for regime in regimes]
    )
    epistemic = _bootstrap_epistemic_calibration(
        labels=labels,
        equity_excess=equity_excess,
        equity_log_excess=equity_log_excess,
        bond_excess=bond_excess,
        rate_surprise=rate_surprise,
        bond_duration=bond_duration,
        transition_counts=counts,
        target_scalars=np.array(
            [
                [
                    regime.equity_risk_premium,
                    regime.bond_term_premium,
                    regime.equity_volatility,
                    regime.bond_residual_volatility,
                    regime.rate_volatility,
                ]
                for regime in regimes
            ]
        ),
        target_initial_probabilities=initial_probabilities,
        samples=epistemic_samples,
        seed=epistemic_seed,
    )
    return FittedRegimeModel(
        regimes=(regimes[0], regimes[1], regimes[2]),
        transition=transition,
        initial_probabilities=initial_probabilities,
        labels=labels,
        stress_log_return_threshold=threshold,
        stress_quantile=stress_quantile,
        bond_duration=bond_duration,
        equity_risk_premium=float(initial_probabilities @ stationary_premiums),
        equity_volatility=float(np.std(equity_excess, ddof=1)),
        initial_real_rate=float(history.real_cash_returns[-1]),
        long_run_real_rate=long_run_rate,
        rate_mean_reversion=mean_reversion,
        rate_volatility=float(np.std(rate_surprise, ddof=2)),
        minimum_real_rate=float(
            np.quantile(history.real_cash_returns, rate_floor_quantile)
        ),
        epistemic=epistemic,
    )


def file_sha256(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _float_literal(value: float) -> str:
    return f"{value:.17g}"


def _tuple_literal(values: FloatArray) -> str:
    return "(" + ", ".join(_float_literal(float(value)) for value in values) + ")"


def _matrix_literal(matrix: FloatArray, *, indent: int) -> str:
    prefix = " " * indent
    rows = "\n".join(f"{prefix}    {_tuple_literal(row)}," for row in matrix)
    return f"(\n{rows}\n{prefix})"


def render_defaults_module(
    fitted: FittedRegimeModel,
    *,
    source_url: str,
    source_sha256: str,
    first_year: int,
    last_year: int,
) -> str:
    """Render the fitted constants as an import-only, zero-calculation module."""

    regime_rows = []
    correlation_rows = []
    for regime in fitted.regimes:
        regime_rows.append(
            "    ("
            + ", ".join(
                _float_literal(value)
                for value in (
                    regime.equity_risk_premium,
                    regime.bond_term_premium,
                    regime.equity_volatility,
                    regime.bond_residual_volatility,
                    regime.rate_volatility,
                    regime.tail_degrees,
                )
            )
            + "),"
        )
        correlation_rows.append(
            "    (\n"
            + "\n".join(
                f"        {_tuple_literal(row)},"
                for row in regime.correlation
            )
            + "\n    ),"
        )
    transition_rows = "\n".join(
        f"    {_tuple_literal(row)}," for row in fitted.transition
    )
    epistemic_scalar_rows = "\n".join(
        f"    {_matrix_literal(matrix, indent=4)},"
        for matrix in fitted.epistemic.regime_scalars
    )
    epistemic_transition_rows = "\n".join(
        f"    {_matrix_literal(matrix, indent=4)},"
        for matrix in fitted.epistemic.transition
    )
    epistemic_initial_rows = "\n".join(
        f"    {_tuple_literal(row)},"
        for row in fitted.epistemic.initial_probabilities
    )
    return f'''"""Generated three-state market calibration; do not edit by hand."""

# ruff: noqa: E501

SOURCE_URL = {source_url!r}
SOURCE_SHA256 = {source_sha256!r}
FIRST_YEAR = {first_year}
LAST_YEAR = {last_year}
STRESS_QUANTILE = {float(fitted.stress_quantile)!r}
BOND_DURATION = {float(fitted.bond_duration)!r}
REGIME_NAMES = {REGIME_NAMES!r}

# equity premium, bond premium, equity log-volatility, bond residual
# volatility, real-rate volatility, Student-t degrees of freedom
REGIME_SCALARS = (
{chr(10).join(regime_rows)}
)

REGIME_CORRELATIONS = (
{chr(10).join(correlation_rows)}
)

TRANSITION = (
{transition_rows}
)
INITIAL_PROBABILITIES = {_tuple_literal(fitted.initial_probabilities)}

# Each path selects one row at inception and retains it for its full horizon.
# Scalar order: equity premium, bond premium, equity log-volatility,
# bond residual volatility, real-rate volatility.
EPISTEMIC_SEED = {fitted.epistemic.seed}
EPISTEMIC_REGIME_SCALARS = (
{epistemic_scalar_rows}
)
EPISTEMIC_TRANSITIONS = (
{epistemic_transition_rows}
)
EPISTEMIC_INITIAL_PROBABILITIES = (
{epistemic_initial_rows}
)

EQUITY_RISK_PREMIUM = {_float_literal(fitted.equity_risk_premium)}
EQUITY_VOLATILITY = {_float_literal(fitted.equity_volatility)}
INITIAL_REAL_RATE = {_float_literal(fitted.initial_real_rate)}
LONG_RUN_REAL_RATE = {_float_literal(fitted.long_run_real_rate)}
RATE_MEAN_REVERSION = {_float_literal(fitted.rate_mean_reversion)}
RATE_VOLATILITY = {_float_literal(fitted.rate_volatility)}
MINIMUM_REAL_RATE = {_float_literal(fitted.minimum_real_rate)}
'''
