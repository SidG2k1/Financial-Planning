"""Regression coverage for finite-value validation and market caching."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from scipy.optimize import brentq

from lifecycle_finance.domain import Allocation, OutcomeType, UtilityAggregation
from lifecycle_finance.markets import CapitalMarketAssumptions
from lifecycle_finance.taxes import TaxPolicy
from lifecycle_finance.utility import (
    IsoelasticCurve,
    LinearCurve,
    SpendingFloorCurve,
    TargetCurve,
    UtilityAddon,
    UtilityOutcome,
    inverse_isoelastic_utility,
    isoelastic_utility,
)


def _capital_market_assumptions(**overrides: object) -> CapitalMarketAssumptions:
    inputs: dict[str, object] = {
        "asset_names": ("equity", "cash"),
        "reference_weights": np.array([0.5, 0.5]),
        "covariance": np.array([[0.04, 0.0], [0.0, 0.0001]]),
        "is_equity": np.array([True, False]),
        "is_global": np.array([False, False]),
        "risk_free_rate": 0.025,
    }
    inputs.update(overrides)
    return CapitalMarketAssumptions(**inputs)  # type: ignore[arg-type]


@pytest.mark.parametrize("component", ["domestic_equity", "global_equity", "bonds", "cash"])
@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_allocation_rejects_nonfinite_components(component: str, nonfinite: float) -> None:
    values = {"domestic_equity": -0.2, "global_equity": 0.3, "bonds": 0.4, "cash": 0.5}
    values[component] = nonfinite

    with pytest.raises(ValueError, match=f"{component} must be finite"):
        Allocation(**values)


def test_allocation_allows_finite_negative_unconstrained_components() -> None:
    allocation = Allocation(-0.2, 0.3, 0.4, 0.5)

    assert allocation.domestic_equity == -0.2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reference_weights", np.array([np.nan, 1.0])),
        ("covariance", np.array([[0.04, np.inf], [0.0, 0.0001]])),
        ("is_equity", np.array([np.nan, 0.0])),
        ("is_global", np.array([0.0, -np.inf])),
    ],
)
def test_capital_market_assumptions_rejects_nonfinite_arrays(
    field: str, value: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match=f"{field} must contain only finite values"):
        _capital_market_assumptions(**{field: value})


@pytest.mark.parametrize("risk_free_rate", [np.nan, np.inf, -np.inf])
def test_capital_market_assumptions_rejects_nonfinite_risk_free_rate(
    risk_free_rate: float,
) -> None:
    with pytest.raises(ValueError, match="risk_free_rate must be finite"):
        _capital_market_assumptions(risk_free_rate=risk_free_rate)


@pytest.mark.parametrize(
    "field",
    [
        "state_ordinary_rate",
        "state_capital_gains_rate",
        "niit_rate",
        "niit_threshold",
        "social_security_wage_base",
        "social_security_payroll_rate",
        "medicare_payroll_rate",
        "additional_medicare_rate",
        "additional_medicare_threshold",
    ],
)
@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_tax_policy_rejects_nonfinite_scalar_fields(
    field: str,
    nonfinite: float,
) -> None:
    with pytest.raises(ValueError, match=field):
        TaxPolicy(**{field: nonfinite})


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("state_ordinary_rate", -0.01),
        ("state_ordinary_rate", 1.01),
        ("state_capital_gains_rate", -0.01),
        ("state_capital_gains_rate", 1.01),
        ("niit_rate", -0.01),
        ("niit_rate", 1.01),
        ("social_security_payroll_rate", -0.01),
        ("social_security_payroll_rate", 1.01),
        ("medicare_payroll_rate", -0.01),
        ("medicare_payroll_rate", 1.01),
        ("additional_medicare_rate", -0.01),
        ("additional_medicare_rate", 1.01),
        ("niit_threshold", -1.0),
        ("social_security_wage_base", -1.0),
        ("additional_medicare_threshold", -1.0),
    ],
)
def test_tax_policy_rejects_scalar_fields_outside_their_valid_ranges(
    field: str,
    invalid: float,
) -> None:
    with pytest.raises(ValueError, match=field):
        TaxPolicy(**{field: invalid})


@pytest.mark.parametrize("elasticity", [np.nan, np.inf, -np.inf])
def test_isoelastic_functions_reject_nonfinite_elasticity(elasticity: float) -> None:
    with pytest.raises(ValueError, match="elasticity must be finite"):
        isoelastic_utility([1.0], elasticity)
    with pytest.raises(ValueError, match="elasticity must be finite"):
        inverse_isoelastic_utility([0.0], elasticity)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_isoelastic_functions_reject_nonfinite_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        isoelastic_utility([value], 0.4)
    with pytest.raises(ValueError, match="finite"):
        inverse_isoelastic_utility([value], 0.4)


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("reference", np.nan),
        ("elasticity", np.inf),
        ("minimum_utility", -np.inf),
    ],
)
def test_isoelastic_curve_rejects_nonfinite_parameters(parameter: str, value: float) -> None:
    inputs = {"reference": 100.0, "elasticity": 0.4, "minimum_utility": -10.0}
    inputs[parameter] = value

    with pytest.raises(ValueError, match=f"{parameter} must be finite"):
        IsoelasticCurve(**inputs)


@pytest.mark.parametrize("method", ["evaluate", "marginal_utility"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_isoelastic_curve_rejects_nonfinite_values(method: str, value: float) -> None:
    curve = IsoelasticCurve(100.0, 0.4)

    with pytest.raises(ValueError, match="value must contain only finite values"):
        getattr(curve, method)([value])


@pytest.mark.parametrize("parameter", ["threshold", "scale", "curvature"])
@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_spending_floor_curve_rejects_nonfinite_parameters(
    parameter: str, nonfinite: float,
) -> None:
    inputs = {"threshold": 40_000.0, "scale": 10_000.0, "curvature": 2.0}
    inputs[parameter] = nonfinite

    with pytest.raises(ValueError, match=f"{parameter} must be finite"):
        SpendingFloorCurve(**inputs)


@pytest.mark.parametrize("parameter", ["target", "tolerance", "curvature"])
@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_target_curve_rejects_nonfinite_parameters(parameter: str, nonfinite: float) -> None:
    inputs = {"target": 65.0, "tolerance": 2.0, "curvature": 2.0}
    inputs[parameter] = nonfinite

    with pytest.raises(ValueError, match=f"{parameter} must be finite"):
        TargetCurve(**inputs)


@pytest.mark.parametrize(
    "curve",
    [
        LinearCurve(),
        SpendingFloorCurve(40_000.0, 10_000.0),
        TargetCurve(65.0, 2.0),
    ],
)
@pytest.mark.parametrize("method", ["evaluate", "marginal_utility"])
@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_utility_curves_reject_nonfinite_values(
    curve: object, method: str, nonfinite: float,
) -> None:
    with pytest.raises(ValueError, match="value must contain only finite values"):
        getattr(curve, method)([nonfinite])


@pytest.mark.parametrize(
    "curve",
    [SpendingFloorCurve(40_000.0, 10_000.0), TargetCurve(65.0, 2.0)],
)
@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_curve_diagnostics_reject_nonfinite_values(
    curve: SpendingFloorCurve | TargetCurve,
    nonfinite: float,
) -> None:
    with pytest.raises(ValueError, match="value must contain only finite values"):
        curve.diagnostic_breach([nonfinite])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("spending", np.array([[np.nan]])),
        ("exposure", np.array([[np.inf]])),
        ("terminal_wealth", -np.inf),
    ],
)
def test_utility_outcome_rejects_nonfinite_core_values(field: str, value: object) -> None:
    inputs: dict[str, object] = {
        "spending": np.array([[50_000.0]]),
        "exposure": np.array([[1.0]]),
        "ages": (45,),
        "terminal_wealth": np.array([100_000.0]),
        "decisions": {},
    }
    inputs[field] = value

    with pytest.raises(ValueError, match=f"{field} must contain only finite values"):
        UtilityOutcome(**inputs)  # type: ignore[arg-type]


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_utility_outcome_rejects_nonfinite_decisions(nonfinite: float) -> None:
    with pytest.raises(ValueError, match=r"decision outcome .*finite"):
        UtilityOutcome(
            spending=np.array([[50_000.0]]),
            exposure=np.array([[1.0]]),
            ages=(45,),
            terminal_wealth=np.array([100_000.0]),
            decisions={OutcomeType.LEVERAGE: nonfinite},
        )


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_utility_addon_rejects_nonfinite_importance(nonfinite: float) -> None:
    with pytest.raises(ValueError, match="importance must be finite"):
        UtilityAddon(
            "spending",
            OutcomeType.SPENDING,
            LinearCurve(),
            importance=nonfinite,
        )


@pytest.mark.parametrize(
    ("minimum_age", "maximum_age", "message"),
    [
        (np.nan, None, "minimum_age must be finite"),
        (None, np.inf, "maximum_age must be finite"),
    ],
)
def test_utility_addon_rejects_nonfinite_age_bounds(
    minimum_age: float | None,
    maximum_age: float | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        UtilityAddon(
            "spending",
            OutcomeType.SPENDING,
            LinearCurve(),
            minimum_age=minimum_age,  # type: ignore[arg-type]
            maximum_age=maximum_age,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "aggregation",
    [UtilityAggregation.WORST, UtilityAggregation.LAST],
)
def test_utility_addon_empty_year_aggregation_returns_zero(
    aggregation: UtilityAggregation,
) -> None:
    outcome = UtilityOutcome(
        spending=np.empty((1, 0)),
        exposure=np.empty((1, 0)),
        ages=(),
        terminal_wealth=np.array([0.0]),
        decisions={},
    )
    addon = UtilityAddon(
        "spending",
        OutcomeType.SPENDING,
        LinearCurve(),
        aggregation=aggregation,
    )

    np.testing.assert_array_equal(addon.score(outcome, np.empty((1, 0))), [0.0])


def test_sigma_sdf_solves_once_per_assumptions_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lifecycle_finance.markets as markets

    calls = 0

    def counted_brentq(function: Callable[[float], float], lower: float, upper: float) -> float:
        nonlocal calls
        calls += 1
        return float(brentq(function, lower, upper))

    monkeypatch.setattr(markets, "brentq", counted_brentq)
    assumptions = _capital_market_assumptions()

    first = assumptions.sigma_sdf()
    second = assumptions.sigma_sdf()

    assert first == second
    assert calls == 1
