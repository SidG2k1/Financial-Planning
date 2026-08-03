from dataclasses import fields, replace

import numpy as np
import pytest

from lifecycle_finance import (
    EpistemicRegimeDistribution,
    MarketModelConfig,
    RegimeModelConfig,
    RegimeSwitchingMarket,
    StochasticMarket,
)


def _absorbing_config(index: int) -> RegimeModelConfig:
    defaults = RegimeModelConfig.defaults()
    probabilities = np.zeros(3)
    probabilities[index] = 1.0
    return replace(
        defaults,
        transition=np.eye(3),
        initial_probabilities=probabilities,
        epistemic=None,
    )


def _regime_market_config(**changes: float) -> MarketModelConfig:
    return replace(RegimeSwitchingMarket().config, **changes)


def test_regime_parameters_reject_invalid_correlation() -> None:
    normal = RegimeModelConfig.defaults().normal
    with pytest.raises(ValueError, match="correlation"):
        replace(normal, correlation=np.eye(2))


def test_regime_configuration_is_immutable_and_validated() -> None:
    config = RegimeModelConfig.defaults()
    assert config.transition.shape == (3, 3)
    assert len(config.regimes) == 3
    assert config.regimes[2] is config.inflation_stress
    assert not config.transition.flags.writeable
    np.testing.assert_allclose(config.transition.sum(axis=1), 1.0)
    assert config.epistemic is not None
    assert config.epistemic.regime_scalars.shape == (128, 3, 5)


def test_market_models_clone_their_own_type() -> None:
    replacement = MarketModelConfig(equity_risk_premium=0.04)
    assert isinstance(StochasticMarket().with_config(replacement), StochasticMarket)
    original = RegimeSwitchingMarket()
    clone = original.with_config(replacement)
    assert isinstance(clone, RegimeSwitchingMarket)
    assert clone.regime_config is original.regime_config


@pytest.mark.parametrize("model", [StochasticMarket(), RegimeSwitchingMarket()])
def test_market_models_apply_config_overrides_without_resetting_other_fields(
    model: StochasticMarket | RegimeSwitchingMarket,
) -> None:
    clone = model.with_config_overrides(
        initial_real_rate=0.015,
        long_run_real_rate=0.015,
    )

    assert type(clone) is type(model)
    assert clone.config.initial_real_rate == 0.015
    assert clone.config.long_run_real_rate == 0.015
    assert clone.config.equity_risk_premium == model.config.equity_risk_premium
    assert clone.config.equity_volatility == model.config.equity_volatility


def test_regime_config_overrides_preserve_model_state() -> None:
    original = RegimeSwitchingMarket().conditioned_on_epistemic_scenario(3)

    clone = original.with_config_overrides(minimum_real_rate=-0.03)

    assert clone.regime_config is original.regime_config
    assert clone.bond_duration == original.bond_duration
    assert clone.epistemic_uncertainty is True
    assert clone.epistemic_scenario == 3


def test_regime_configuration_rejects_probability_overrides_with_stale_epistemic_draws() -> None:
    defaults = RegimeModelConfig.defaults()

    with pytest.raises(ValueError, match=r"transition.*epistemic"):
        replace(defaults, transition=np.eye(3))
    with pytest.raises(ValueError, match=r"initial_probabilities.*epistemic"):
        replace(
            defaults,
            initial_probabilities=np.array([1.0, 0.0, 0.0]),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stock_tail_degrees", 10.0),
        ("stock_bond_correlation", 0.5),
    ],
)
def test_regime_market_rejects_unrouted_market_overrides(
    field: str,
    value: float,
) -> None:
    with pytest.raises(ValueError, match=field):
        RegimeSwitchingMarket().with_config_overrides(**{field: value})


def test_regime_rate_volatility_override_scales_fitted_rate_risk() -> None:
    model = RegimeSwitchingMarket().with_config_overrides(rate_volatility=0.0)

    generated = model.generate(
        paths=2_000,
        horizon=4,
        seed=79,
        antithetic=False,
    )

    expected = np.maximum(
        model.config.initial_real_rate
        + model.config.rate_mean_reversion
        * (model.config.long_run_real_rate - model.config.initial_real_rate),
        model.config.minimum_real_rate,
    )
    np.testing.assert_allclose(generated.real_rates[:, 1], expected)


def test_regime_market_rejects_invalid_epistemic_scenario() -> None:
    with pytest.raises(ValueError, match="out of range"):
        RegimeSwitchingMarket(epistemic_scenario=128)
    with pytest.raises(ValueError, match="requires epistemic_uncertainty"):
        RegimeSwitchingMarket(
            epistemic_uncertainty=False,
            epistemic_scenario=0,
        )


def test_fitted_regime_defaults_do_not_change_legacy_market_defaults() -> None:
    legacy = StochasticMarket()
    regime = RegimeSwitchingMarket()

    assert legacy.config == MarketModelConfig()
    assert legacy.config.equity_risk_premium == 0.025
    assert legacy.config.equity_volatility == 0.10
    assert regime.config.equity_risk_premium > legacy.config.equity_risk_premium
    assert regime.config.equity_volatility > legacy.config.equity_volatility


def test_simulation_reexports_stochastic_market_for_compatibility() -> None:
    from lifecycle_finance.simulation import StochasticMarket as CompatibilityMarket

    assert CompatibilityMarket is StochasticMarket


def test_singular_positive_semidefinite_correlation_is_accepted() -> None:
    normal = RegimeModelConfig.defaults().normal
    singular = replace(normal, correlation=np.ones((3, 3)))

    assert not singular._root.flags.writeable
    np.testing.assert_allclose(singular._root @ singular._root.T, singular.correlation)


def test_positive_definite_correlation_precomputes_its_cholesky_factor() -> None:
    normal = RegimeModelConfig.defaults().normal

    np.testing.assert_allclose(
        normal._root,
        np.linalg.cholesky(normal.correlation),
    )


def test_near_singular_positive_definite_correlation_uses_cholesky() -> None:
    minimum_eigenvalue = 5e-11
    off_diagonal = 1.0 - minimum_eigenvalue
    correlation = np.full((3, 3), off_diagonal)
    np.fill_diagonal(correlation, 1.0)

    parameters = replace(
        RegimeModelConfig.defaults().normal,
        correlation=correlation,
    )

    assert np.min(np.linalg.eigvalsh(parameters.correlation)) == pytest.approx(
        minimum_eigenvalue,
        rel=1e-5,
    )
    np.testing.assert_array_equal(
        parameters._root,
        np.linalg.cholesky(parameters.correlation),
    )


def test_slightly_indefinite_correlation_is_rejected() -> None:
    minimum_eigenvalue = -5e-11
    off_diagonal = (minimum_eigenvalue - 1.0) / 2.0
    correlation = np.full((3, 3), off_diagonal)
    np.fill_diagonal(correlation, 1.0)

    with pytest.raises(ValueError, match="positive semidefinite"):
        replace(
            RegimeModelConfig.defaults().normal,
            correlation=correlation,
        )


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("equity_risk_premium", np.nan),
        ("equity_risk_premium", np.inf),
        ("bond_term_premium", np.nan),
        ("bond_term_premium", -np.inf),
        ("equity_volatility", np.nan),
        ("equity_volatility", np.inf),
        ("bond_residual_volatility", np.nan),
        ("bond_residual_volatility", -np.inf),
        ("rate_volatility", np.nan),
        ("rate_volatility", np.inf),
        ("tail_degrees", np.nan),
        ("tail_degrees", -np.inf),
    ],
)
def test_regime_parameters_reject_nonfinite_scalars(field: str, invalid: float) -> None:
    normal = RegimeModelConfig.defaults().normal
    with pytest.raises(ValueError, match=field):
        replace(normal, **{field: invalid})


@pytest.mark.parametrize(
    "field",
    [
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
    ],
)
def test_market_model_config_rejects_nonfinite_scalar_fields(field: str) -> None:
    with pytest.raises(ValueError, match=f"{field} must be finite"):
        MarketModelConfig(**{field: np.nan})


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("initial_real_rate", np.inf),
        ("initial_real_rate", -np.inf),
        ("long_run_real_rate", np.inf),
        ("long_run_real_rate", -np.inf),
        ("minimum_real_rate", np.inf),
        ("minimum_real_rate", -np.inf),
    ],
)
def test_market_model_config_rejects_infinite_rate_fields(field: str, invalid: float) -> None:
    with pytest.raises(ValueError, match=f"{field} must be finite"):
        MarketModelConfig(**{field: invalid})


def test_market_model_config_allows_finite_negative_rates() -> None:
    config = MarketModelConfig(
        initial_real_rate=-0.01,
        long_run_real_rate=-0.005,
        minimum_real_rate=-0.05,
    )

    assert config.initial_real_rate == -0.01
    assert config.long_run_real_rate == -0.005
    assert config.minimum_real_rate == -0.05


@pytest.mark.parametrize("bond_duration", [0.0, -1.0, np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("model", [StochasticMarket, RegimeSwitchingMarket])
def test_market_models_reject_nonpositive_or_nonfinite_bond_duration(
    model: type[StochasticMarket] | type[RegimeSwitchingMarket], bond_duration: float
) -> None:
    with pytest.raises(ValueError, match="bond_duration"):
        model(bond_duration=bond_duration)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        (
            "transition",
            np.array([[-0.1, 1.1, 0.0], [0.4, 0.6, 0.0], [0.0, 0.0, 1.0]]),
            "transition",
        ),
        (
            "initial_probabilities",
            np.array([1.1, -0.1, 0.0]),
            "initial_probabilities",
        ),
        (
            "transition",
            np.array([[0.9, 0.0, 0.0], [0.4, 0.6, 0.0], [0.0, 0.0, 1.0]]),
            "transition rows",
        ),
        (
            "initial_probabilities",
            np.array([0.8, 0.1, 0.0]),
            "initial_probabilities must sum",
        ),
    ],
)
def test_regime_configuration_rejects_invalid_probabilities(
    field: str, value: np.ndarray, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        replace(RegimeModelConfig.defaults(), **{field: value})


@pytest.mark.parametrize(
    ("correlation", "match"),
    [
        (
            np.array([[1.0, 0.2, 0.0], [0.1, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            "symmetric",
        ),
        (
            np.array([[0.9, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            "unit diagonal",
        ),
        (
            np.array([[1.0, -0.9, -0.9], [-0.9, 1.0, -0.9], [-0.9, -0.9, 1.0]]),
            "positive semidefinite",
        ),
    ],
)
def test_regime_parameters_reject_invalid_correlation_properties(
    correlation: np.ndarray, match: str
) -> None:
    normal = RegimeModelConfig.defaults().normal
    with pytest.raises(ValueError, match=match):
        replace(normal, correlation=correlation)


def test_regime_configuration_copies_and_freezes_input_arrays() -> None:
    defaults = RegimeModelConfig.defaults()
    correlation = defaults.normal.correlation.copy()
    transition = defaults.transition.copy()
    initial_probabilities = defaults.initial_probabilities.copy()
    config = RegimeModelConfig(
        replace(defaults.normal, correlation=correlation),
        defaults.stress,
        defaults.inflation_stress,
        transition,
        initial_probabilities,
        None,
    )

    correlation[0, 1] = 0.75
    transition[0, 0] = 0.75
    initial_probabilities[0] = 0.75

    assert config.normal.correlation[0, 1] != 0.75
    assert config.transition[0, 0] == defaults.transition[0, 0]
    assert config.initial_probabilities[0] == defaults.initial_probabilities[0]
    assert not config.normal.correlation.flags.writeable
    assert not config.transition.flags.writeable
    assert not config.initial_probabilities.flags.writeable


def test_epistemic_distribution_is_validated_copied_and_frozen() -> None:
    scalars = np.ones((2, 3, 5))
    transition = np.repeat(np.eye(3)[np.newaxis, :, :], 2, axis=0)
    probabilities = np.repeat(
        np.array([[1.0, 0.0, 0.0]]),
        2,
        axis=0,
    )
    distribution = EpistemicRegimeDistribution(
        scalars,
        transition,
        probabilities,
        np.eye(3),
        np.array([1.0, 0.0, 0.0]),
    )

    scalars[0, 0, 0] = 99.0
    transition[0, 0, 0] = 0.5
    probabilities[0, 0] = 0.5

    assert distribution.regime_scalars[0, 0, 0] == 1.0
    assert distribution.transition[0, 0, 0] == 1.0
    assert distribution.initial_probabilities[0, 0] == 1.0
    assert not distribution.regime_scalars.flags.writeable
    assert not distribution.transition.flags.writeable
    assert not distribution.initial_probabilities.flags.writeable


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("regime_scalars", np.ones((2, 2, 5)), "regime_scalars"),
        ("transition", np.ones((2, 2, 2)), "transition"),
        ("initial_probabilities", np.ones((2, 2)), "initial_probabilities"),
    ],
)
def test_epistemic_distribution_rejects_wrong_shapes(
    field: str,
    value: np.ndarray,
    match: str,
) -> None:
    values = {
        "regime_scalars": np.ones((2, 3, 5)),
        "transition": np.repeat(np.eye(3)[np.newaxis, :, :], 2, axis=0),
        "initial_probabilities": np.repeat(
            np.array([[1.0, 0.0, 0.0]]),
            2,
            axis=0,
        ),
    }
    values[field] = value

    with pytest.raises(ValueError, match=match):
        EpistemicRegimeDistribution(**values)


@pytest.mark.parametrize("model", [StochasticMarket(), RegimeSwitchingMarket()])
def test_market_model_is_seeded_finite_and_rate_floored(
    model: StochasticMarket | RegimeSwitchingMarket,
) -> None:
    first = model.generate(paths=257, horizon=30, seed=8)
    second = model.generate(paths=257, horizon=30, seed=8)
    for item in fields(type(first)):
        left = getattr(first, item.name)
        right = getattr(second, item.name)
        np.testing.assert_array_equal(left, right)
        assert np.all(np.isfinite(left))
    assert np.min(first.real_rates) >= model.config.minimum_real_rate


@pytest.mark.parametrize("model", [StochasticMarket(), RegimeSwitchingMarket()])
def test_market_path_outputs_retain_the_existing_shapes(
    model: StochasticMarket | RegimeSwitchingMarket,
) -> None:
    generated = model.generate(paths=7, horizon=4, seed=19)

    for item in fields(type(generated)):
        values = getattr(generated, item.name)
        assert values.shape == (7, 4)
        assert np.all(np.isfinite(values))


@pytest.mark.parametrize("model", [StochasticMarket(), RegimeSwitchingMarket()])
@pytest.mark.parametrize(
    ("paths", "horizon", "match"),
    [
        (0, 2, "paths must be positive"),
        (-1, 2, "paths must be positive"),
        (2, 0, "horizon must be positive"),
        (2, -1, "horizon must be positive"),
    ],
)
def test_market_path_generation_rejects_invalid_dimensions(
    model: StochasticMarket | RegimeSwitchingMarket,
    paths: int,
    horizon: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        model.generate(paths=paths, horizon=horizon, seed=19)


@pytest.mark.parametrize("model_type", [StochasticMarket, RegimeSwitchingMarket])
def test_market_models_enforce_the_rate_floor_on_the_initial_state(
    model_type: type[StochasticMarket] | type[RegimeSwitchingMarket],
) -> None:
    config = MarketModelConfig(
        initial_real_rate=-0.05,
        minimum_real_rate=-0.02,
    )
    generated = model_type(config).generate(paths=3, horizon=1, seed=23)

    np.testing.assert_array_equal(
        generated.real_rates[:, 0],
        np.full(3, config.minimum_real_rate),
    )
    np.testing.assert_array_equal(
        generated.cash_returns[:, 0],
        np.full(3, config.minimum_real_rate),
    )


def test_legacy_unlevered_returns_use_minimal_limited_liability_floor() -> None:
    config = MarketModelConfig(
        equity_volatility=10.0,
        stock_tail_degrees=2.1,
        volatility_of_volatility=0.0,
        rate_volatility=10.0,
        minimum_real_rate=-1.0,
    )
    generated = StochasticMarket(config).generate(
        paths=100,
        horizon=2,
        seed=107,
        antithetic=False,
    )
    minimum_return = np.nextafter(-1.0, 0.0)

    assert np.min(generated.equity_returns) == minimum_return
    assert np.min(generated.bond_returns) == minimum_return


@pytest.mark.parametrize("configured_correlation", [-0.20, 0.50])
def test_legacy_market_realizes_configured_stock_bond_correlation(
    configured_correlation: float,
) -> None:
    config = MarketModelConfig(
        stock_bond_correlation=configured_correlation,
        initial_real_rate=0.02,
        long_run_real_rate=0.02,
        rate_mean_reversion=0.0,
        rate_volatility=0.01,
        minimum_real_rate=-1.0,
        volatility_of_volatility=0.0,
    )
    generated = StochasticMarket(config).generate(
        paths=50_000,
        horizon=1,
        seed=31,
        antithetic=False,
    )

    realized = np.corrcoef(
        generated.equity_returns[:, 0],
        generated.bond_returns[:, 0],
    )[0, 1]

    assert realized == pytest.approx(configured_correlation, abs=0.02)


def test_odd_antithetic_paths_keep_complete_pairs_and_truncate_the_last_pair() -> None:
    regime_config = _absorbing_config(0)
    parameters = regime_config.normal
    base = _regime_market_config(
        volatility_of_volatility=0.0,
        initial_real_rate=0.20,
        long_run_real_rate=0.20,
        minimum_real_rate=-0.50,
    )
    model = RegimeSwitchingMarket(base, regime_config)
    odd = model.generate(paths=5, horizon=2, seed=29)
    even = model.generate(paths=6, horizon=2, seed=29)

    for item in fields(type(odd)):
        np.testing.assert_array_equal(
            getattr(odd, item.name),
            getattr(even, item.name)[:5],
        )

    rate_change = odd.real_rates[:, 1] - odd.real_rates[:, 0]
    continuous_shocks = np.column_stack(
        (
            np.arctanh(
                (1.0 + odd.equity_returns[:, 0])
                / (
                    1.0
                    + odd.cash_returns[:, 0]
                    + parameters.equity_risk_premium
                )
                - 1.0
            )
            / parameters.equity_volatility,
            (
                odd.bond_returns[:, 0]
                - odd.cash_returns[:, 0]
                - parameters.bond_term_premium
                + model.bond_duration * rate_change
            )
            / parameters.bond_residual_volatility,
            rate_change / parameters.rate_volatility,
        )
    )
    np.testing.assert_allclose(
        continuous_shocks[:2],
        -continuous_shocks[3:],
        atol=1e-14,
    )


def test_each_path_keeps_one_epistemic_parameter_draw_for_its_full_horizon() -> None:
    defaults = RegimeModelConfig.defaults()
    zero_risk = tuple(
        replace(
            parameters,
            equity_volatility=0.0,
            bond_residual_volatility=0.0,
            rate_volatility=0.0,
        )
        for parameters in defaults.regimes
    )
    scalar_draws = np.zeros((2, 3, 5))
    scalar_draws[0, :, 0] = 0.01
    scalar_draws[1, :, 0] = 0.10
    epistemic = EpistemicRegimeDistribution(
        scalar_draws,
        np.repeat(np.eye(3)[np.newaxis, :, :], 2, axis=0),
        np.repeat(np.array([[1.0, 0.0, 0.0]]), 2, axis=0),
        np.eye(3),
        np.array([1.0, 0.0, 0.0]),
    )
    regime_config = RegimeModelConfig(
        zero_risk[0],
        zero_risk[1],
        zero_risk[2],
        np.eye(3),
        np.array([1.0, 0.0, 0.0]),
        epistemic,
    )
    base = _regime_market_config(
        volatility_of_volatility=0.0,
        initial_real_rate=0.02,
        long_run_real_rate=0.02,
        minimum_real_rate=-1.0,
    )
    generated = RegimeSwitchingMarket(base, regime_config).generate(
        paths=100,
        horizon=5,
        seed=101,
    )
    equity_excess = generated.equity_returns - generated.cash_returns

    np.testing.assert_allclose(
        equity_excess,
        np.repeat(equity_excess[:, :1], 5, axis=1),
        atol=1e-15,
    )
    assert set(np.round(equity_excess[:, 0], 12)) == {0.01, 0.10}
    np.testing.assert_array_equal(equity_excess[:50], equity_excess[50:])


def test_conditioned_epistemic_scenario_is_shared_by_every_path() -> None:
    defaults = RegimeModelConfig.defaults()
    zero_risk = tuple(
        replace(
            parameters,
            equity_volatility=0.0,
            bond_residual_volatility=0.0,
            rate_volatility=0.0,
        )
        for parameters in defaults.regimes
    )
    scalar_draws = np.zeros((2, 3, 5))
    scalar_draws[0, :, 0] = 0.01
    scalar_draws[1, :, 0] = 0.10
    epistemic = EpistemicRegimeDistribution(
        scalar_draws,
        np.repeat(np.eye(3)[np.newaxis, :, :], 2, axis=0),
        np.repeat(np.array([[1.0, 0.0, 0.0]]), 2, axis=0),
        np.eye(3),
        np.array([1.0, 0.0, 0.0]),
    )
    regime_config = RegimeModelConfig(
        zero_risk[0],
        zero_risk[1],
        zero_risk[2],
        np.eye(3),
        np.array([1.0, 0.0, 0.0]),
        epistemic,
    )
    base = _regime_market_config(
        volatility_of_volatility=0.0,
        initial_real_rate=0.02,
        long_run_real_rate=0.02,
        minimum_real_rate=-1.0,
    )
    model = RegimeSwitchingMarket(
        base,
        regime_config,
    ).conditioned_on_epistemic_scenario(1)
    generated = model.generate(paths=100, horizon=5, seed=101)
    equity_excess = generated.equity_returns - generated.cash_returns

    np.testing.assert_allclose(equity_excess, 0.10, atol=1e-15)


def test_equity_log_return_mapping_never_falls_below_minus_one() -> None:
    defaults = RegimeModelConfig.defaults()
    extreme = replace(
        defaults.normal,
        equity_volatility=10.0,
        tail_degrees=2.1,
    )
    regime_config = replace(
        defaults,
        normal=extreme,
        transition=np.eye(3),
        initial_probabilities=np.array([1.0, 0.0, 0.0]),
        epistemic=None,
    )
    base = _regime_market_config(
        volatility_of_volatility=0.0,
        initial_real_rate=0.02,
        long_run_real_rate=0.02,
    )
    generated = RegimeSwitchingMarket(base, regime_config).generate(
        paths=500_000,
        horizon=2,
        seed=103,
        antithetic=False,
    )

    assert np.min(generated.equity_returns) > -1.0
    assert np.all(np.isfinite(generated.equity_returns))


def test_regime_bond_returns_never_fall_below_minus_one() -> None:
    defaults = RegimeModelConfig.defaults()
    extreme = replace(
        defaults.normal,
        bond_residual_volatility=10.0,
        rate_volatility=10.0,
        tail_degrees=2.1,
    )
    regime_config = replace(
        defaults,
        normal=extreme,
        transition=np.eye(3),
        initial_probabilities=np.array([1.0, 0.0, 0.0]),
        epistemic=None,
    )
    base = _regime_market_config(
        volatility_of_volatility=0.0,
        initial_real_rate=0.02,
        long_run_real_rate=0.02,
        minimum_real_rate=-1.0,
    )
    generated = RegimeSwitchingMarket(base, regime_config).generate(
        paths=500_000,
        horizon=2,
        seed=104,
        antithetic=False,
    )

    assert np.min(generated.bond_returns) > -1.0
    assert np.all(np.isfinite(generated.bond_returns))


def test_returns_use_the_current_regime_before_transitioning_for_the_next_year() -> None:
    defaults = RegimeModelConfig.defaults()
    normal = replace(
        defaults.normal,
        equity_volatility=0.0,
        bond_residual_volatility=0.0,
        rate_volatility=0.0,
    )
    stress = replace(
        defaults.stress,
        equity_volatility=0.0,
        bond_residual_volatility=0.0,
        rate_volatility=0.0,
    )
    inflation_stress = replace(
        defaults.inflation_stress,
        equity_volatility=0.0,
        bond_residual_volatility=0.0,
        rate_volatility=0.0,
    )
    switch_after_year_zero = RegimeModelConfig(
        normal,
        stress,
        inflation_stress,
        np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        np.array([1.0, 0.0, 0.0]),
        None,
    )
    base = _regime_market_config(
        volatility_of_volatility=0.0,
        initial_real_rate=0.02,
        long_run_real_rate=0.02,
    )
    generated = RegimeSwitchingMarket(base, switch_after_year_zero).generate(
        paths=4,
        horizon=3,
        seed=41,
        antithetic=False,
    )

    equity_excess = generated.equity_returns - generated.cash_returns
    bond_excess = generated.bond_returns - generated.cash_returns
    np.testing.assert_allclose(
        equity_excess,
        np.array(
            [
                [
                    normal.equity_risk_premium,
                    stress.equity_risk_premium,
                    inflation_stress.equity_risk_premium,
                ]
            ]
            * 4
        ),
    )
    np.testing.assert_allclose(
        bond_excess,
        np.array(
            [
                [
                    normal.bond_term_premium,
                    stress.bond_term_premium,
                    inflation_stress.bond_term_premium,
                ]
            ]
            * 4
        ),
    )


@pytest.mark.parametrize("regime_index", [0, 1, 2])
def test_equity_premium_sweep_shifts_each_regime_return(regime_index: int) -> None:
    premium_delta = 0.0125
    regime_config = _absorbing_config(regime_index)
    base_config = _regime_market_config(
        volatility_of_volatility=0.0,
        initial_real_rate=0.20,
        long_run_real_rate=0.20,
        minimum_real_rate=-1.0,
    )
    swept_config = replace(
        base_config,
        equity_risk_premium=base_config.equity_risk_premium + premium_delta,
    )

    baseline = RegimeSwitchingMarket(base_config, regime_config).generate(
        paths=257,
        horizon=1,
        seed=47,
        antithetic=False,
    )
    swept = RegimeSwitchingMarket(swept_config, regime_config).generate(
        paths=257,
        horizon=1,
        seed=47,
        antithetic=False,
    )

    parameters = regime_config.regimes[regime_index]
    baseline_location = (
        1.0 + baseline.cash_returns + parameters.equity_risk_premium
    )
    swept_location = baseline_location + premium_delta
    np.testing.assert_allclose(
        (1.0 + swept.equity_returns) / (1.0 + baseline.equity_returns),
        swept_location / baseline_location,
        atol=1e-15,
    )


@pytest.mark.parametrize("regime_index", [0, 1, 2])
def test_equity_volatility_sweep_scales_each_regime_shock(regime_index: int) -> None:
    volatility_ratio = 1.4
    regime_config = _absorbing_config(regime_index)
    parameters = regime_config.regimes[regime_index]
    base_config = _regime_market_config(
        volatility_of_volatility=0.0,
        initial_real_rate=0.20,
        long_run_real_rate=0.20,
        minimum_real_rate=-1.0,
    )
    swept_config = replace(
        base_config,
        equity_volatility=base_config.equity_volatility * volatility_ratio,
    )

    baseline = RegimeSwitchingMarket(base_config, regime_config).generate(
        paths=257,
        horizon=1,
        seed=53,
        antithetic=False,
    )
    swept = RegimeSwitchingMarket(swept_config, regime_config).generate(
        paths=257,
        horizon=1,
        seed=53,
        antithetic=False,
    )

    baseline_shock = np.arctanh(
        (1.0 + baseline.equity_returns)
        / (
            1.0
            + baseline.cash_returns
            + parameters.equity_risk_premium
        )
        - 1.0
    )
    swept_shock = np.arctanh(
        (1.0 + swept.equity_returns)
        / (
            1.0
            + swept.cash_returns
            + parameters.equity_risk_premium
        )
        - 1.0
    )
    np.testing.assert_allclose(
        swept_shock,
        baseline_shock * volatility_ratio,
        atol=1e-15,
    )


def test_predictable_rate_mean_reversion_does_not_create_a_bond_windfall() -> None:
    defaults = RegimeModelConfig.defaults()
    normal = replace(
        defaults.normal,
        equity_volatility=0.0,
        bond_residual_volatility=0.0,
        rate_volatility=0.0,
    )
    regime_config = replace(
        defaults,
        normal=normal,
        transition=np.eye(3),
        initial_probabilities=np.array([1.0, 0.0, 0.0]),
        epistemic=None,
    )
    base = _regime_market_config(
        volatility_of_volatility=0.0,
        initial_real_rate=-0.015,
        long_run_real_rate=-1.0,
        rate_mean_reversion=0.15,
        minimum_real_rate=-0.02,
    )
    model = RegimeSwitchingMarket(base, regime_config, bond_duration=6.0)
    generated = model.generate(paths=3, horizon=2, seed=43, antithetic=False)
    expected_first_bond_return = base.initial_real_rate + normal.bond_term_premium

    np.testing.assert_allclose(generated.real_rates[:, 1], base.minimum_real_rate)
    np.testing.assert_allclose(generated.bond_returns[:, 0], expected_first_bond_return)


def test_rate_floor_limits_the_duration_effect_to_the_realized_rate_innovation() -> None:
    defaults = RegimeModelConfig.defaults()
    normal = replace(
        defaults.normal,
        equity_volatility=0.0,
        bond_residual_volatility=0.0,
        rate_volatility=0.10,
    )
    regime_config = replace(
        defaults,
        normal=normal,
        transition=np.eye(3),
        initial_probabilities=np.array([1.0, 0.0, 0.0]),
        epistemic=None,
    )
    base = _regime_market_config(
        volatility_of_volatility=0.0,
        initial_real_rate=-0.015,
        long_run_real_rate=-0.015,
        rate_mean_reversion=0.15,
        minimum_real_rate=-0.02,
    )
    model = RegimeSwitchingMarket(base, regime_config, bond_duration=6.0)

    generated = model.generate(
        paths=20_000,
        horizon=2,
        seed=83,
        antithetic=False,
    )

    floor_hit = generated.real_rates[:, 1] == base.minimum_real_rate
    assert np.any(floor_hit)
    realized_innovation = base.minimum_real_rate - base.initial_real_rate
    expected_bond_return = (
        base.initial_real_rate
        + normal.bond_term_premium
        - model.bond_duration * realized_innovation
    )
    np.testing.assert_allclose(
        generated.bond_returns[floor_hit, 0],
        expected_bond_return,
    )


def test_stochastic_volatility_is_stationary_and_variance_normalized() -> None:
    legacy_config = replace(
        MarketModelConfig(),
        equity_risk_premium=0.0,
        equity_volatility=0.10,
        stock_bond_correlation=0.0,
        initial_real_rate=0.0,
        long_run_real_rate=0.0,
        rate_volatility=0.0,
        minimum_real_rate=-1.0,
    )
    legacy = StochasticMarket(legacy_config).generate(
        paths=200_000,
        horizon=15,
        seed=89,
        antithetic=False,
    )
    assert np.std(legacy.equity_returns[:, 0]) == pytest.approx(0.10, rel=0.025)
    assert np.std(legacy.equity_returns[:, -1]) == pytest.approx(0.10, rel=0.025)

    regime_config = _absorbing_config(0)
    parameters = regime_config.normal
    regime_market_config = _regime_market_config(
        initial_real_rate=0.0,
        long_run_real_rate=0.0,
        rate_volatility=0.0,
        minimum_real_rate=-1.0,
    )
    regime = RegimeSwitchingMarket(
        regime_market_config,
        regime_config,
    ).generate(
        paths=200_000,
        horizon=15,
        seed=97,
        antithetic=False,
    )
    location = (
        1.0
        + regime.cash_returns
        + parameters.equity_risk_premium
    )
    normalized_latent = (
        np.arctanh((1.0 + regime.equity_returns) / location - 1.0)
        / parameters.equity_volatility
    )
    assert np.std(normalized_latent[:, 0]) == pytest.approx(1.0, rel=0.025)
    assert np.std(normalized_latent[:, -1]) == pytest.approx(1.0, rel=0.025)


def test_default_regime_market_realizes_its_fitted_equity_volatility() -> None:
    model = RegimeSwitchingMarket()

    generated = model.generate(
        paths=200_000,
        horizon=15,
        seed=101,
        antithetic=False,
    )

    realized = float(np.std(generated.equity_returns))
    assert realized == pytest.approx(model.config.equity_volatility, rel=0.05)
    assert np.std(generated.equity_returns[:, 0]) == pytest.approx(
        np.std(generated.equity_returns[:, -1]),
        rel=0.04,
    )


def test_fitted_stress_states_have_negative_equity_premiums_and_more_risk() -> None:
    normal = _absorbing_config(0)
    parameters = normal.regimes

    assert parameters[0].equity_risk_premium > 0.0
    assert parameters[1].equity_risk_premium < 0.0
    assert parameters[2].equity_risk_premium < 0.0
    assert parameters[1].equity_volatility > parameters[0].equity_volatility
    assert parameters[2].equity_volatility > parameters[0].equity_volatility


@pytest.mark.parametrize("regime_index", [0, 1, 2])
def test_absorbing_regime_reproduces_innovation_scale_and_correlation(
    regime_index: int,
) -> None:
    regime_config = _absorbing_config(regime_index)
    parameters = regime_config.regimes[regime_index]
    base = _regime_market_config(
        volatility_of_volatility=0.0,
        initial_real_rate=0.20,
        long_run_real_rate=0.20,
        minimum_real_rate=-0.50,
    )
    model = RegimeSwitchingMarket(base, regime_config)
    generated = model.generate(paths=200_000, horizon=2, seed=31, antithetic=False)

    rate_change = generated.real_rates[:, 1] - generated.real_rates[:, 0]
    innovations = np.column_stack(
        (
            np.arctanh(
                (1.0 + generated.equity_returns[:, 0])
                / (
                    1.0
                    + generated.cash_returns[:, 0]
                    + parameters.equity_risk_premium
                )
                - 1.0
            )
            / parameters.equity_volatility,
            (
                generated.bond_returns[:, 0]
                - generated.cash_returns[:, 0]
                - parameters.bond_term_premium
                + model.bond_duration * rate_change
            )
            / parameters.bond_residual_volatility,
            rate_change / parameters.rate_volatility,
        )
    )

    equity_excess = generated.equity_returns[:, 0] - generated.cash_returns[:, 0]
    bond_excess = generated.bond_returns[:, 0] - generated.cash_returns[:, 0]
    assert np.mean(equity_excess) == pytest.approx(
        parameters.equity_risk_premium,
        abs=0.0075,
    )
    assert np.mean(bond_excess) == pytest.approx(
        parameters.bond_term_premium,
        abs=0.0075,
    )
    np.testing.assert_allclose(np.std(innovations, axis=0), np.ones(3), rtol=0.05)
    np.testing.assert_allclose(
        np.corrcoef(innovations, rowvar=False),
        parameters.correlation,
        atol=0.03,
    )


def test_growth_and_inflation_stress_have_opposite_stock_bond_correlations() -> None:
    base = _regime_market_config(
        volatility_of_volatility=0.0,
        initial_real_rate=0.02,
        long_run_real_rate=0.02,
        minimum_real_rate=-1.0,
    )
    correlations = []
    for regime_index in (1, 2):
        generated = RegimeSwitchingMarket(
            base,
            _absorbing_config(regime_index),
        ).generate(
            paths=300_000,
            horizon=1,
            seed=107,
            antithetic=False,
        )
        equity_excess = generated.equity_returns[:, 0] - generated.cash_returns[:, 0]
        bond_excess = generated.bond_returns[:, 0] - generated.cash_returns[:, 0]
        correlations.append(float(np.corrcoef(equity_excess, bond_excess)[0, 1]))

    assert correlations[0] < -0.05
    assert correlations[1] > 0.02


def test_default_market_has_squared_equity_risk_persistence() -> None:
    generated = RegimeSwitchingMarket().generate(
        paths=200_000,
        horizon=2,
        seed=32,
        antithetic=False,
    )
    equity_excess = generated.equity_returns - generated.cash_returns

    assert np.corrcoef(equity_excess[:, 0] ** 2, equity_excess[:, 1] ** 2)[0, 1] > 0.02


def test_default_market_has_real_rate_level_persistence() -> None:
    generated = RegimeSwitchingMarket().generate(
        paths=200_000,
        horizon=3,
        seed=33,
        antithetic=False,
    )

    assert np.corrcoef(generated.real_rates[:, 1], generated.real_rates[:, 2])[0, 1] > 0.25


def test_default_market_has_joint_equity_bond_tail_dependence() -> None:
    generated = RegimeSwitchingMarket().generate(
        paths=400_000,
        horizon=1,
        seed=34,
        antithetic=False,
    )
    absolute_equity = np.abs(generated.equity_returns[:, 0])
    absolute_bonds = np.abs(generated.bond_returns[:, 0])
    equity_threshold = np.quantile(absolute_equity, 0.95)
    bond_threshold = np.quantile(absolute_bonds, 0.95)
    joint_exceedance = np.mean(
        (absolute_equity > equity_threshold) & (absolute_bonds > bond_threshold)
    )

    assert joint_exceedance > 0.0026


def test_default_stationary_expected_excess_returns_match_base_assumptions() -> None:
    model = RegimeSwitchingMarket()
    generated = model.generate(
        paths=200_000,
        horizon=30,
        seed=35,
        antithetic=False,
    )
    equity_excess = generated.equity_returns - generated.cash_returns
    bond_excess = generated.bond_returns - generated.cash_returns

    assert np.mean(equity_excess) == pytest.approx(
        model.config.equity_risk_premium,
        abs=0.0125,
    )
    fitted_bond_premium = sum(
        probability * parameters.bond_term_premium
        for probability, parameters in zip(
            model.regime_config.initial_probabilities,
            model.regime_config.regimes,
            strict=True,
        )
    )
    assert np.mean(bond_excess[:, 10:]) == pytest.approx(
        fitted_bond_premium,
        abs=0.0125,
    )
