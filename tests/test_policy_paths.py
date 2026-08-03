from __future__ import annotations

import numpy as np
import pytest

from lifecycle_finance.domain import OutcomeType, Person, Preferences, UtilityAggregation
from lifecycle_finance.policy_paths import (
    PolicyPathContext,
    PolicyPathDecision,
    PolicyPathEvaluator,
)
from lifecycle_finance.return_models import MarketPaths
from lifecycle_finance.utility import (
    IsoelasticCurve,
    SpendingFloorCurve,
    TargetCurve,
    UtilityAddon,
    UtilityModel,
    UtilityOutcome,
)


def flat_market(*, paths: int, horizon: int) -> MarketPaths:
    zeros = np.zeros((paths, horizon))
    return MarketPaths(zeros, zeros.copy(), zeros.copy(), zeros.copy())


def market_with_equity_return(*, equity_return: float) -> MarketPaths:
    equity = np.full((1, 2), equity_return)
    zeros = np.zeros((1, 2))
    return MarketPaths(equity, zeros.copy(), zeros.copy(), zeros.copy())


class FixedPolicy:
    def decide(self, context: PolicyPathContext) -> PolicyPathDecision:
        return PolicyPathDecision(
            consumption=60.0,
            target_total_equity=0.5,
            external_income=100.0,
            restricted_vesting=20.0,
        )


def evaluator() -> PolicyPathEvaluator:
    return PolicyPathEvaluator()


def test_policy_paths_apply_income_consumption_vesting_and_total_equity() -> None:
    market = flat_market(paths=2, horizon=2)
    result = evaluator().evaluate(
        market,
        FixedPolicy(),
        candidate_count=1,
        ages=(40.0, 41.0, 42.0),
        exposure=np.ones(3),
        initial_liquid_wealth=100.0,
    )

    np.testing.assert_allclose(result.invested_liquid_wealth[0, :, 0], 140.0)
    np.testing.assert_allclose(result.invested_restricted_equity[0, :, 0], 20.0)
    np.testing.assert_allclose(result.actual_total_equity[0, :, 0], 0.5)


def test_policy_paths_release_preserves_total_wealth_and_makes_equity_liquid() -> None:
    class ReleasePolicy:
        def decide(self, context: PolicyPathContext) -> PolicyPathDecision:
            return PolicyPathDecision(
                consumption=0.0,
                target_total_equity=0.0,
                restricted_release=30.0 if context.year == 0 else 0.0,
            )

    result = evaluator().evaluate(
        flat_market(paths=1, horizon=2),
        ReleasePolicy(),
        candidate_count=1,
        ages=(40.0, 41.0, 42.0),
        exposure=np.ones(3),
        initial_liquid_wealth=100.0,
        initial_restricted_equity=40.0,
    )

    np.testing.assert_allclose(result.beginning_total_wealth[0, :, 0], 140.0)
    np.testing.assert_allclose(result.invested_liquid_wealth[0, :, 0], 130.0)
    np.testing.assert_allclose(result.invested_restricted_equity[0, :, 0], 10.0)


def test_policy_paths_funds_ordinary_consumption_before_event_spending() -> None:
    class SpendingPolicy:
        def decide(self, context: PolicyPathContext) -> PolicyPathDecision:
            return PolicyPathDecision(
                consumption=80.0,
                target_total_equity=0.0,
                event_spending=80.0,
                event_outcome=OutcomeType.WEDDING_SPEND,
            )

    result = evaluator().evaluate(
        flat_market(paths=1, horizon=2),
        SpendingPolicy(),
        candidate_count=1,
        ages=(40.0, 41.0, 42.0),
        exposure=np.ones(3),
        initial_liquid_wealth=100.0,
    )

    np.testing.assert_allclose(result.consumption[0, :, 0], 80.0)
    np.testing.assert_allclose(result.event_spending[0, :, 0], 20.0)
    assert not result.ordinary_shortfall[0, :, 0].any()
    assert result.event_shortfall[0, :, 0].all()


def test_policy_paths_mark_shortfalls_when_requested_spending_exceeds_liquidity() -> None:
    class ShortfallPolicy:
        def decide(self, context: PolicyPathContext) -> PolicyPathDecision:
            return PolicyPathDecision(
                consumption=70.0,
                target_total_equity=0.0,
                event_spending=10.0,
                event_outcome=OutcomeType.WEDDING_SPEND,
            )

    result = evaluator().evaluate(
        flat_market(paths=1, horizon=2),
        ShortfallPolicy(),
        candidate_count=1,
        ages=(40.0, 41.0, 42.0),
        exposure=np.ones(3),
        initial_liquid_wealth=50.0,
    )

    np.testing.assert_allclose(result.consumption[0, :, 0], 50.0)
    np.testing.assert_allclose(result.event_spending[0, :, 0], 0.0)
    assert result.ordinary_shortfall[0, :, 0].all()
    assert result.event_shortfall[0, :, 0].all()


def test_policy_paths_does_not_clip_valid_near_total_loss_returns() -> None:
    class EquityPolicy:
        def decide(self, context: PolicyPathContext) -> PolicyPathDecision:
            return PolicyPathDecision(consumption=0.0, target_total_equity=1.0)

    result = evaluator().evaluate(
        market_with_equity_return(equity_return=-0.99),
        EquityPolicy(),
        candidate_count=1,
        ages=(40.0, 41.0, 42.0),
        exposure=np.ones(3),
        initial_liquid_wealth=100.0,
    )

    np.testing.assert_allclose(result.beginning_total_wealth[0, :, 1], 1.0)


def test_policy_paths_apply_bond_return_to_non_equity_liquid_wealth() -> None:
    class BondPolicy:
        def decide(self, context: PolicyPathContext) -> PolicyPathDecision:
            return PolicyPathDecision(consumption=0.0, target_total_equity=0.0)

    zeros = np.zeros((1, 2))
    market = MarketPaths(
        equity_returns=zeros.copy(),
        bond_returns=np.array([[0.10, 0.0]]),
        cash_returns=np.array([[-0.50, 0.0]]),
        real_rates=zeros.copy(),
    )
    result = evaluator().evaluate(
        market,
        BondPolicy(),
        candidate_count=1,
        ages=(40.0, 41.0, 42.0),
        exposure=np.ones(3),
        initial_liquid_wealth=100.0,
    )

    np.testing.assert_allclose(result.beginning_total_wealth[0, :, 1], 110.0)


def test_policy_paths_reject_release_above_restricted_equity() -> None:
    class InvalidReleasePolicy:
        def decide(self, context: PolicyPathContext) -> PolicyPathDecision:
            return PolicyPathDecision(
                consumption=0.0,
                target_total_equity=0.0,
                restricted_release=21.0,
            )

    with np.testing.assert_raises(ValueError):
        evaluator().evaluate(
            flat_market(paths=1, horizon=2),
            InvalidReleasePolicy(),
            candidate_count=1,
            ages=(40.0, 41.0, 42.0),
            exposure=np.ones(3),
            initial_liquid_wealth=100.0,
            initial_restricted_equity=20.0,
        )


def test_policy_paths_rejects_ambiguous_equal_length_vector() -> None:
    class ZeroPolicy:
        def decide(self, context: PolicyPathContext) -> PolicyPathDecision:
            return PolicyPathDecision(consumption=0.0, target_total_equity=0.0)

    with np.testing.assert_raises_regex(ValueError, "ambiguous"):
        evaluator().evaluate(
            flat_market(paths=2, horizon=2),
            ZeroPolicy(),
            candidate_count=2,
            ages=(40.0, 41.0, 42.0),
            exposure=np.ones(3),
            initial_liquid_wealth=np.array([100.0, 200.0]),
        )


def test_policy_paths_accepts_explicit_path_vector_when_counts_equal() -> None:
    class ZeroPolicy:
        def decide(self, context: PolicyPathContext) -> PolicyPathDecision:
            return PolicyPathDecision(consumption=0.0, target_total_equity=0.0)

    result = evaluator().evaluate(
        flat_market(paths=2, horizon=2),
        ZeroPolicy(),
        candidate_count=2,
        ages=(40.0, 41.0, 42.0),
        exposure=np.ones(3),
        initial_liquid_wealth=np.array([[100.0, 200.0]]),
    )

    np.testing.assert_allclose(
        result.beginning_total_wealth[:, :, 0],
        np.array([[100.0, 200.0], [100.0, 200.0]]),
    )


def test_policy_paths_accepts_explicit_candidate_vector_when_counts_equal() -> None:
    class ZeroPolicy:
        def decide(self, context: PolicyPathContext) -> PolicyPathDecision:
            return PolicyPathDecision(consumption=0.0, target_total_equity=0.0)

    result = evaluator().evaluate(
        flat_market(paths=2, horizon=2),
        ZeroPolicy(),
        candidate_count=2,
        ages=(40.0, 41.0, 42.0),
        exposure=np.ones(3),
        initial_liquid_wealth=np.array([[100.0], [200.0]]),
    )

    np.testing.assert_allclose(
        result.beginning_total_wealth[:, :, 0],
        np.array([[100.0, 100.0], [200.0, 200.0]]),
    )


def test_policy_paths_scores_library_utility_components() -> None:
    utility_model = UtilityModel(
        person=Person(current_age=40, retirement_age=41, maximum_age=42),
        preferences=Preferences(
            time_preference=0.1,
            vitality_peak_age=100,
            spending_floor=0.0,
            bequest_strength=0.0,
        ),
        addons=(
            UtilityAddon(
                "consumption",
                OutcomeType.SPENDING,
                IsoelasticCurve(100.0, 1.0),
            ),
            UtilityAddon(
                "spending_floor",
                OutcomeType.SPENDING,
                SpendingFloorCurve(100.0, 25.0),
            ),
            UtilityAddon(
                "retirement_freedom",
                OutcomeType.RETIRED,
                TargetCurve(1.0, 0.25),
                minimum_age=41,
            ),
            UtilityAddon(
                "wedding",
                OutcomeType.WEDDING_SPEND,
                TargetCurve(30.0, 10.0),
                aggregation=UtilityAggregation.DISCOUNTED_SUM,
            ),
        ),
    )

    class UtilityPolicy:
        def decide(self, context: PolicyPathContext) -> PolicyPathDecision:
            outcomes = {
                OutcomeType.RETIRED: np.array([[0.0], [1.0]]),
                OutcomeType.WORKING: np.array([[1.0], [0.0]]),
            }
            if context.year == 0:
                return PolicyPathDecision(
                    consumption=np.array([[100.0], [125.0]]),
                    target_total_equity=0.0,
                    external_income=200.0,
                    outcomes=outcomes,
                )
            return PolicyPathDecision(
                consumption=np.array([[75.0], [100.0]]),
                target_total_equity=0.0,
                external_income=200.0,
                event_spending=np.array([[30.0], [20.0]]),
                event_outcome=OutcomeType.WEDDING_SPEND,
                outcomes=outcomes,
            )

    result = PolicyPathEvaluator(utility_model).evaluate(
        flat_market(paths=2, horizon=2),
        UtilityPolicy(),
        candidate_count=2,
        ages=(40.5, 41.5, 42.5),
        exposure=(1.0, 0.5, 0.0),
        initial_liquid_wealth=0.0,
    )

    utility_outcome = UtilityOutcome(
        spending=result.consumption.reshape(4, 2),
        exposure=np.broadcast_to(np.array([1.0, 0.5]), (2, 2, 2)).reshape(4, 2),
        ages=(40.5, 41.5),
        terminal_wealth=np.zeros(4),
        decisions={
            outcome: values.reshape(4, 2)
            for outcome, values in result.decision_outcomes.items()
        },
    )
    expected_components = utility_model.decompose(utility_outcome)

    for name, expected in expected_components.items():
        np.testing.assert_allclose(
            result.utility_component_scores[name],
            expected.reshape(2, 2),
        )
    np.testing.assert_allclose(
        result.utility_scores,
        np.sum(np.stack(tuple(expected_components.values())), axis=0).reshape(2, 2),
    )
    np.testing.assert_array_equal(
        result.preference_breaches["retirement_freedom"],
        np.array(
            [
                [[False, True], [False, True]],
                [[False, False], [False, False]],
            ]
        ),
    )


@pytest.mark.parametrize(
    ("decision", "message"),
    [
        (
            PolicyPathDecision(
                consumption=np.nan,
                target_total_equity=0.0,
            ),
            "consumption must be finite",
        ),
        (
            PolicyPathDecision(
                consumption=0.0,
                target_total_equity=0.0,
                external_income=-1.0,
            ),
            "external_income must be nonnegative",
        ),
        (
            PolicyPathDecision(
                consumption=0.0,
                target_total_equity=0.0,
                event_spending=-1.0,
            ),
            "event_spending must be nonnegative",
        ),
        (
            PolicyPathDecision(
                consumption=0.0,
                target_total_equity=1.1,
            ),
            "target_total_equity must be between 0 and 1",
        ),
    ],
)
def test_policy_paths_rejects_invalid_decision_values(
    decision: PolicyPathDecision,
    message: str,
) -> None:
    class StaticPolicy:
        def decide(self, context: PolicyPathContext) -> PolicyPathDecision:
            return decision

    with pytest.raises(ValueError, match=message):
        evaluator().evaluate(
            flat_market(paths=1, horizon=2),
            StaticPolicy(),
            candidate_count=1,
            ages=(40.0, 41.0, 42.0),
            exposure=np.ones(3),
            initial_liquid_wealth=100.0,
        )


def test_policy_paths_rejects_misaligned_market_arrays() -> None:
    market = MarketPaths(
        equity_returns=np.zeros((1, 2)),
        bond_returns=np.zeros((2, 2)),
        cash_returns=np.zeros((1, 2)),
        real_rates=np.zeros((1, 2)),
    )

    with pytest.raises(ValueError, match="bond_returns must align with equity_returns"):
        evaluator().evaluate(
            market,
            FixedPolicy(),
            candidate_count=1,
            ages=(40.0, 41.0, 42.0),
            exposure=np.ones(3),
            initial_liquid_wealth=100.0,
        )


def test_policy_paths_rejects_exposure_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match="exposure must be between 0 and 1"):
        evaluator().evaluate(
            flat_market(paths=1, horizon=2),
            FixedPolicy(),
            candidate_count=1,
            ages=(40.0, 41.0, 42.0),
            exposure=(1.0, 1.1, 0.0),
            initial_liquid_wealth=100.0,
        )


def test_policy_paths_requires_an_event_outcome_for_event_spending() -> None:
    class MissingEventOutcomePolicy:
        def decide(self, context: PolicyPathContext) -> PolicyPathDecision:
            return PolicyPathDecision(
                consumption=0.0,
                target_total_equity=0.0,
                event_spending=10.0,
            )

    with pytest.raises(
        ValueError,
        match="event_outcome is required when event_spending is nonzero",
    ):
        evaluator().evaluate(
            flat_market(paths=1, horizon=2),
            MissingEventOutcomePolicy(),
            candidate_count=1,
            ages=(40.0, 41.0, 42.0),
            exposure=np.ones(3),
            initial_liquid_wealth=100.0,
        )


def test_policy_paths_rejects_event_outcome_collisions() -> None:
    class DuplicateEventOutcomePolicy:
        def decide(self, context: PolicyPathContext) -> PolicyPathDecision:
            return PolicyPathDecision(
                consumption=0.0,
                target_total_equity=0.0,
                event_spending=10.0,
                event_outcome=OutcomeType.WEDDING_SPEND,
                outcomes={OutcomeType.WEDDING_SPEND: 5.0},
            )

    with pytest.raises(ValueError, match="event_outcome cannot duplicate outcomes"):
        evaluator().evaluate(
            flat_market(paths=1, horizon=2),
            DuplicateEventOutcomePolicy(),
            candidate_count=1,
            ages=(40.0, 41.0, 42.0),
            exposure=np.ones(3),
            initial_liquid_wealth=100.0,
        )
