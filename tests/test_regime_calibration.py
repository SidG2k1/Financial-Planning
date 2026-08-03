from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from lifecycle_finance import calibrated_regime_defaults
from lifecycle_finance.regime_calibration import (
    FittedRegimeModel,
    HistoricalReturns,
    _validate_stationary_vector,
    fit_regime_model,
    render_defaults_module,
)


def _synthetic_history() -> HistoricalReturns:
    years = np.arange(1977, 2026)
    index = np.arange(len(years))
    cash = 0.01 + 0.004 * np.sin(index / 3.0)
    equity = 0.08 + 0.02 * np.sin(index)
    bonds = 0.035 + 0.01 * np.cos(index / 2.0)
    labels = np.tile(np.array([0, 0, 0, 1, 0, 0, 0, 2]), 6)
    growth = np.flatnonzero(labels == 1)
    inflation = np.flatnonzero(labels == 2)
    equity[growth] = -0.18 + 0.015 * np.sin(growth)
    bonds[growth] = 0.08 + 0.01 * np.cos(growth)
    equity[inflation] = -0.24 + 0.015 * np.cos(inflation)
    bonds[inflation] = -0.08 + 0.01 * np.sin(inflation)
    return HistoricalReturns(
        years=years.astype(np.int64),
        real_equity_returns=equity,
        real_cash_returns=cash,
        real_bond_returns=bonds,
    )


def test_offline_fit_recovers_three_states_and_centered_epistemic_draws() -> None:
    fitted = fit_regime_model(
        _synthetic_history(),
        epistemic_samples=16,
        epistemic_seed=11,
    )

    np.testing.assert_array_equal(np.bincount(fitted.labels), np.array([36, 6, 6]))
    np.testing.assert_allclose(fitted.transition.sum(axis=1), 1.0)
    np.testing.assert_allclose(fitted.initial_probabilities.sum(), 1.0)
    assert fitted.regimes[0].equity_risk_premium > 0.0
    assert fitted.regimes[1].equity_risk_premium < 0.0
    assert fitted.regimes[2].equity_risk_premium < 0.0
    assert fitted.regimes[1].bond_term_premium > 0.0
    assert fitted.regimes[2].bond_term_premium < 0.0
    assert fitted.epistemic.regime_scalars.shape == (16, 3, 5)
    assert fitted.epistemic.transition.shape == (16, 3, 3)
    assert fitted.epistemic.initial_probabilities.shape == (16, 3)
    target_scalars = np.array(
        [
            [
                regime.equity_risk_premium,
                regime.bond_term_premium,
                regime.equity_volatility,
                regime.bond_residual_volatility,
                regime.rate_volatility,
            ]
            for regime in fitted.regimes
        ]
    )
    sampled_means = np.mean(fitted.epistemic.regime_scalars, axis=0)
    np.testing.assert_allclose(sampled_means[:, 2:], target_scalars[:, 2:])
    for column in (0, 1):
        np.testing.assert_allclose(
            sampled_means[:, column] - target_scalars[:, column],
            np.full(
                3,
                sampled_means[0, column] - target_scalars[0, column],
            ),
        )
        sampled_unconditional = np.mean(
            np.sum(
                fitted.epistemic.initial_probabilities
                * fitted.epistemic.regime_scalars[:, :, column],
                axis=1,
            )
        )
        target_unconditional = (
            fitted.initial_probabilities @ target_scalars[:, column]
        )
        np.testing.assert_allclose(
            sampled_unconditional,
            target_unconditional,
        )
    assert np.std(fitted.epistemic.regime_scalars[:, :, 0]) > 0.0
    assert np.std(fitted.epistemic.transition[:, 0, 0]) > 0.0


def test_generated_defaults_module_contains_provenance_and_uncertainty() -> None:
    history = _synthetic_history()
    fitted = fit_regime_model(
        history,
        epistemic_samples=4,
        epistemic_seed=13,
    )
    rendered = render_defaults_module(
        fitted,
        source_url="https://example.test/returns.xls",
        source_sha256="abc123",
        first_year=int(history.years[0]),
        last_year=int(history.years[-1]),
    )

    assert "https://example.test/returns.xls" in rendered
    assert "SOURCE_SHA256 = 'abc123'" in rendered
    assert "REGIME_NAMES = ('normal', 'growth_stress', 'inflation_stress')" in rendered
    assert "EPISTEMIC_REGIME_SCALARS" in rendered
    assert "EPISTEMIC_TRANSITIONS" in rendered
    assert "STRESS_QUANTILE = 0.25" in rendered
    assert "BOND_DURATION = 6.0" in rendered


def test_fitted_model_preserves_legacy_positional_construction() -> None:
    fitted = fit_regime_model(
        _synthetic_history(),
        epistemic_samples=4,
        epistemic_seed=13,
    )

    reconstructed = FittedRegimeModel(
        fitted.regimes,
        fitted.transition,
        fitted.initial_probabilities,
        fitted.labels,
        fitted.stress_log_return_threshold,
        fitted.equity_risk_premium,
        fitted.equity_volatility,
        fitted.initial_real_rate,
        fitted.long_run_real_rate,
        fitted.rate_mean_reversion,
        fitted.rate_volatility,
        fitted.minimum_real_rate,
        fitted.epistemic,
    )

    assert reconstructed.stress_quantile == 0.25
    assert reconstructed.bond_duration == 6.0


def test_fitted_and_rendered_defaults_preserve_calibration_provenance() -> None:
    history = _synthetic_history()
    fitted = fit_regime_model(
        history,
        bond_duration=7.5,
        stress_quantile=0.375,
        epistemic_samples=4,
        epistemic_seed=13,
    )
    rendered = render_defaults_module(
        fitted,
        source_url="https://example.test/returns.xls",
        source_sha256="abc123",
        first_year=int(history.years[0]),
        last_year=int(history.years[-1]),
    )

    assert fitted.bond_duration == 7.5
    assert fitted.stress_quantile == 0.375
    assert "STRESS_QUANTILE = 0.375" in rendered
    assert "BOND_DURATION = 7.5" in rendered


def test_checked_in_calibration_identifies_its_authoritative_sample() -> None:
    assert calibrated_regime_defaults.FIRST_YEAR == 1928
    assert calibrated_regime_defaults.LAST_YEAR == 2025
    assert calibrated_regime_defaults.SOURCE_SHA256 == (
        "12430ac0d0e762b8d0b0a1246089111ec18245278f224e7f6e9925496ca901d8"
    )
    assert len(calibrated_regime_defaults.EPISTEMIC_REGIME_SCALARS) == 128


def test_calibration_cli_writes_and_checks_csv_output(tmp_path: Path) -> None:
    history = _synthetic_history()
    source = tmp_path / "history.csv"
    output = tmp_path / "generated.py"
    rows = [
        "year,real_equity_return,real_cash_return,real_bond_return\n",
        *[
            f"{year},{equity},{cash},{bond}\n"
            for year, equity, cash, bond in zip(
                history.years,
                history.real_equity_returns,
                history.real_cash_returns,
                history.real_bond_returns,
                strict=True,
            )
        ],
    ]
    source.write_text("".join(rows))

    script = Path(__file__).parents[1] / "tools" / "calibrate_regime_model.py"

    def run(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(script), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )

    assert run(str(source), "--output", str(output)).returncode == 0
    assert run(str(source), "--output", str(output), "--check").returncode == 0
    output.write_text(output.read_text() + "\n")
    assert run(str(source), "--output", str(output), "--check").returncode == 1


def test_stationary_vector_rejects_a_genuinely_negative_candidate() -> None:
    with pytest.raises(ValueError):
        _validate_stationary_vector(np.array([0.7, -0.5, 1.8]))


def test_stationary_vector_clips_negligible_numerical_noise() -> None:
    result = _validate_stationary_vector(np.array([0.5, 0.5 + 1e-13, -1e-13]))

    assert np.all(result >= 0.0)
    assert result.sum() == pytest.approx(1.0)


def test_stationary_vector_flips_a_negated_eigenvector() -> None:
    result = _validate_stationary_vector(np.array([-0.2, -0.3, -0.5]))

    np.testing.assert_allclose(result, [0.2, 0.3, 0.5])


def test_unconditional_equity_volatility_uses_the_unbiased_estimator() -> None:
    history = _synthetic_history()
    fitted = fit_regime_model(history, epistemic_samples=4, epistemic_seed=11)

    equity = history.real_equity_returns[:-1]
    cash = history.real_cash_returns[:-1]
    expected = np.std(equity - cash, ddof=1)

    assert fitted.equity_volatility == pytest.approx(expected)
