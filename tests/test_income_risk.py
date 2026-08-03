import numpy as np
import pytest

from lifecycle_finance.income_risk import (
    IncomeRiskContext,
    IncomeRiskPaths,
    IncomeRiskState,
    IncomeRiskStep,
    PersistentDisplacementIncomeRisk,
    TransitoryMarketJobLoss,
    generate_income_risk_paths,
)


class MalformedPathCountIncomeRisk:
    """Return internally consistent protocol records with the wrong path count."""

    def __init__(self, malformed_stage: str) -> None:
        self.malformed_stage = malformed_stage

    def initial_state(self, paths: int) -> IncomeRiskState:
        output_paths = 1 if self.malformed_stage == "initial state" else paths
        return IncomeRiskState(
            displaced=np.zeros(output_paths, dtype=bool),
            years_since_displacement=np.zeros(output_paths, dtype=int),
        )

    def transition(
        self,
        context: IncomeRiskContext,
        state: IncomeRiskState,
    ) -> IncomeRiskStep:
        output_paths = (
            1
            if self.malformed_stage == "transition step"
            else context.deterministic_income.size
        )
        next_state = IncomeRiskState(
            displaced=np.zeros(output_paths, dtype=bool),
            years_since_displacement=np.zeros(output_paths, dtype=int),
        )
        return IncomeRiskStep(
            state=next_state,
            realized_income=np.full(output_paths, 100.0),
            income_fraction=np.ones(output_paths),
            vesting_eligible=np.ones(output_paths, dtype=bool),
        )


def _context(**changes: object) -> IncomeRiskContext:
    values: dict[str, object] = {
        "year": 0,
        "deterministic_income": np.array([100.0, 100.0]),
        "working": np.array([True, True]),
    }
    values.update(changes)
    paths = np.asarray(values["deterministic_income"]).shape[0]
    values.setdefault("current_excess_return", np.zeros(paths))
    values.setdefault("lagged_excess_return", np.zeros(paths))
    values.setdefault("random_uniform", np.full(paths, 0.5))
    return IncomeRiskContext(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("baseline_probability", np.nan),
        ("baseline_probability", -0.01),
        ("baseline_probability", 1.01),
        ("market_sensitivity", np.inf),
        ("income_fraction", np.nan),
        ("income_fraction", -0.01),
        ("income_fraction", 1.01),
        ("probability_cap", np.nan),
        ("probability_cap", -0.01),
        ("probability_cap", 1.01),
    ],
)
def test_transitory_model_rejects_invalid_probability_and_fraction_inputs(
    field: str,
    value: float,
) -> None:
    values = {
        "baseline_probability": 0.10,
        "market_sensitivity": 2.0,
        "income_fraction": 0.25,
        "probability_cap": 0.50,
    }
    values[field] = value

    with pytest.raises(ValueError, match=field):
        TransitoryMarketJobLoss(**values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("deterministic_income", np.array([100.0, np.nan])),
        ("current_excess_return", np.array([0.0, np.inf])),
        ("lagged_excess_return", np.array([0.0, -np.inf])),
        ("random_uniform", np.array([-0.01, 0.5])),
        ("random_uniform", np.array([0.5, 1.01])),
        ("working", np.array([True])),
    ],
)
def test_context_rejects_nonfinite_out_of_range_and_mismatched_path_arrays(
    field: str,
    value: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        _context(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("realized_income", np.array([100.0, 100.0])),
        ("income_fraction", np.ones((2, 3))),
        ("displaced", np.zeros((3, 2), dtype=bool)),
        ("vesting_eligible", np.ones((3, 2), dtype=bool)),
    ],
)
def test_paths_reject_non_2d_and_mismatched_arrays(field: str, value: np.ndarray) -> None:
    values: dict[str, np.ndarray] = {
        "realized_income": np.full((2, 2), 100.0),
        "income_fraction": np.ones((2, 2)),
        "displaced": np.zeros((2, 2), dtype=bool),
        "vesting_eligible": np.ones((2, 2), dtype=bool),
    }
    values[field] = value

    with pytest.raises(ValueError):
        IncomeRiskPaths(**values)


def test_transitory_job_loss_uses_current_market_excess() -> None:
    model = TransitoryMarketJobLoss(
        baseline_probability=0.10,
        market_sensitivity=2.0,
        income_fraction=0.25,
        probability_cap=0.50,
    )
    context = IncomeRiskContext(
        year=1,
        deterministic_income=np.array([100.0, 100.0]),
        working=np.array([True, True]),
        current_excess_return=np.array([-0.20, 0.10]),
        lagged_excess_return=np.zeros(2),
        random_uniform=np.array([0.11, 0.11]),
    )
    step = model.transition(context, model.initial_state(2))
    np.testing.assert_allclose(step.realized_income, [25.0, 100.0])
    np.testing.assert_array_equal(step.vesting_eligible, [True, True])


def _persistent_model(
    fractions: tuple[float, ...] = (0.25, 0.70, 0.75, 0.80, 0.85),
) -> PersistentDisplacementIncomeRisk:
    return PersistentDisplacementIncomeRisk(
        baseline_probability=0.10,
        market_sensitivity=2.0,
        probability_cap=0.50,
        income_fractions_after_displacement=fractions,
    )


def test_persistent_displacement_uses_lagged_not_current_market_excess() -> None:
    model = _persistent_model()
    context = _context(
        year=1,
        current_excess_return=np.array([0.90, -0.90]),
        lagged_excess_return=np.array([-0.20, 0.00]),
        random_uniform=np.array([0.11, 0.11]),
    )

    step = model.transition(context, model.initial_state(2))

    np.testing.assert_array_equal(step.state.displaced, [True, False])


def test_persistent_displacement_occurs_once_and_advances_recovery_each_working_year() -> None:
    model = _persistent_model()
    state = model.initial_state(1)
    fractions: list[float] = []
    for year in range(6):
        step = model.transition(
            _context(
                year=year,
                deterministic_income=np.array([100.0]),
                working=np.array([True]),
                current_excess_return=np.array([0.90]),
                lagged_excess_return=np.array([0.90]),
                random_uniform=np.array([0.00]),
            ),
            state,
        )
        state = step.state
        fractions.append(float(step.income_fraction[0]))

    np.testing.assert_allclose(fractions, [0.25, 0.70, 0.75, 0.80, 0.85, 0.85])
    np.testing.assert_array_equal(state.displaced, [True])
    np.testing.assert_array_equal(state.years_since_displacement, [5])


def test_persistent_displacement_supports_severe_recovery_schedule() -> None:
    model = _persistent_model((0.00, 0.25, 0.50, 0.5375, 0.575, 0.6125, 0.65))
    state = model.initial_state(1)
    fractions: list[float] = []
    for year in range(7):
        step = model.transition(
            _context(
                year=year,
                deterministic_income=np.array([100.0]),
                working=np.array([True]),
                random_uniform=np.array([0.00]),
            ),
            state,
        )
        state = step.state
        fractions.append(float(step.income_fraction[0]))

    np.testing.assert_allclose(fractions, [0.00, 0.25, 0.50, 0.5375, 0.575, 0.6125, 0.65])


def test_persistent_displacement_revokes_vesting_in_loss_year_permanently() -> None:
    model = _persistent_model()
    state = model.initial_state(1)
    vesting: list[bool] = []
    for year, working in enumerate((True, True, False)):
        step = model.transition(
            _context(
                year=year,
                deterministic_income=np.array([100.0]),
                working=np.array([working]),
                random_uniform=np.array([0.00]),
            ),
            state,
        )
        state = step.state
        vesting.append(bool(step.vesting_eligible[0]))

    assert vesting == [False, False, False]


def test_income_risk_runner_generates_vectorized_persistent_displacement_paths() -> None:
    uniforms = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.20, 0.00, 0.00],
        ]
    )
    original_uniforms = uniforms.copy()

    generated = generate_income_risk_paths(
        _persistent_model((0.25, 0.70)),
        deterministic_income=np.array([100.0, 100.0, 100.0]),
        equity_returns=np.array(
            [
                [0.00, -0.20, 0.10],
                [0.00, 0.10, 0.10],
            ]
        ),
        real_rates=np.zeros((2, 3)),
        equity_risk_premium=0.00,
        working_years=3,
        random_uniforms=uniforms,
    )

    np.testing.assert_allclose(
        generated.realized_income,
        [[25.0, 70.0, 70.0], [100.0, 25.0, 70.0]],
    )
    np.testing.assert_allclose(
        generated.income_fraction,
        [[0.25, 0.70, 0.70], [1.00, 0.25, 0.70]],
    )
    np.testing.assert_array_equal(
        generated.displaced,
        [[True, True, True], [False, True, True]],
    )
    np.testing.assert_array_equal(
        generated.vesting_eligible,
        [[False, False, False], [True, False, False]],
    )
    np.testing.assert_array_equal(uniforms, original_uniforms)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("equity_returns", np.zeros(2)),
        ("real_rates", np.zeros((2, 2))),
        ("random_uniforms", np.full((2, 3), 1.01)),
    ],
)
def test_income_risk_runner_rejects_invalid_matrices(
    field: str,
    value: np.ndarray,
) -> None:
    values: dict[str, object] = {
        "deterministic_income": np.array([100.0, 100.0, 100.0]),
        "equity_returns": np.zeros((2, 3)),
        "real_rates": np.zeros((2, 3)),
        "equity_risk_premium": 0.00,
        "working_years": 3,
        "random_uniforms": np.full((2, 3), 0.5),
    }
    values[field] = value

    with pytest.raises(ValueError):
        generate_income_risk_paths(_persistent_model(), **values)  # type: ignore[arg-type]


def test_income_risk_runner_rejects_negative_working_duration() -> None:
    with pytest.raises(ValueError, match="working_years"):
        generate_income_risk_paths(
            _persistent_model(),
            deterministic_income=np.array([100.0]),
            equity_returns=np.zeros((1, 1)),
            real_rates=np.zeros((1, 1)),
            equity_risk_premium=0.00,
            working_years=-1,
            random_uniforms=np.full((1, 1), 0.5),
        )


@pytest.mark.parametrize("malformed_stage", ["initial state", "transition step"])
def test_income_risk_runner_rejects_model_output_with_wrong_path_count(
    malformed_stage: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=rf"income risk {malformed_stage} must contain exactly 2 paths",
    ):
        generate_income_risk_paths(
            MalformedPathCountIncomeRisk(malformed_stage),
            deterministic_income=np.array([100.0]),
            equity_returns=np.zeros((2, 1)),
            real_rates=np.zeros((2, 1)),
            equity_risk_premium=0.00,
            working_years=1,
            random_uniforms=np.full((2, 1), 0.5),
        )
