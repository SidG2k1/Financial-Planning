"""Preference calibration from explicit indifference trade-offs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import cast

import numpy as np
from scipy.optimize import brentq

from .utility import (
    DifferentiableUtilityCurve,
    UtilityAddon,
    UtilityModel,
    UtilityOutcome,
)


@dataclass(frozen=True, slots=True)
class ImportanceCalibration:
    addon: str
    importance: float
    base_utility_difference: float
    unweighted_addon_difference: float


@dataclass(frozen=True, slots=True)
class UtilityCalibrator:
    """Translate indifference statements into auditable utility weights."""

    model: UtilityModel

    def calibrate_importance(
        self,
        addon_name: str,
        option_a: UtilityOutcome,
        option_b: UtilityOutcome,
    ) -> ImportanceCalibration:
        """Find the nonnegative importance making A and B indifferent."""
        matches = [addon for addon in self.model.addons if addon.name == addon_name]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one add-on named {addon_name!r}")
        addon = matches[0]
        base_model = UtilityModel(
            self.model.person,
            self.model.preferences,
            tuple(component for component in self.model.addons if component.name != addon_name),
        )
        base_difference = float(
            np.mean(base_model.score(option_a)) - np.mean(base_model.score(option_b))
        )
        unit_addon = replace(addon, importance=1.0)
        score_a = unit_addon.score(option_a, self.model.weights(option_a))
        score_b = unit_addon.score(option_b, self.model.weights(option_b))
        addon_difference = float(np.mean(score_a) - np.mean(score_b))
        if np.isclose(addon_difference, 0.0):
            raise ValueError("options do not identify this add-on's importance")
        importance = -base_difference / addon_difference
        if importance < 0:
            raise ValueError(
                "indifference implies a negative importance; reverse or revise the stated trade-off"
            )
        return ImportanceCalibration(
            addon=addon_name,
            importance=importance,
            base_utility_difference=base_difference,
            unweighted_addon_difference=addon_difference,
        )

    def with_calibrated_importance(
        self,
        calibration: ImportanceCalibration,
    ) -> UtilityModel:
        return UtilityModel(
            self.model.person,
            self.model.preferences,
            tuple(
                replace(addon, importance=calibration.importance)
                if addon.name == calibration.addon
                else addon
                for addon in self.model.addons
            ),
        )

    def equivalent_constant_spending(
        self,
        target: UtilityOutcome,
        *,
        lower: float = 1.0,
        upper: float = 10_000_000.0,
    ) -> float:
        """Constant annual spending with the same mean composite utility."""
        if lower <= 0 or upper <= lower:
            raise ValueError("equivalent-spending bounds must satisfy 0 < lower < upper")
        target_score = float(np.mean(self.model.score(target)))

        def difference(spending: float) -> float:
            candidate = replace(
                target,
                spending=np.full_like(target.spending, spending),
            )
            return float(np.mean(self.model.score(candidate))) - target_score

        low_score = difference(lower)
        high_score = difference(upper)
        if low_score > 0 or high_score < 0:
            raise ValueError("equivalent spending is outside the supplied bounds")
        return float(brentq(difference, lower, upper))

    def marginal_rate_of_substitution(
        self,
        first: UtilityAddon,
        first_value: float,
        second: UtilityAddon,
        second_value: float,
    ) -> float:
        """Negative indifference-curve slope; the conventional positive MRS is its magnitude.

        The magnitude is positive when the two marginal utilities share a sign.
        """
        first_curve = cast(DifferentiableUtilityCurve, first.curve)
        second_curve = cast(DifferentiableUtilityCurve, second.curve)
        if not callable(getattr(first_curve, "marginal_utility", None)):
            raise TypeError(f"{first.name!r} does not expose marginal utility")
        if not callable(getattr(second_curve, "marginal_utility", None)):
            raise TypeError(f"{second.name!r} does not expose marginal utility")
        first_marginal = first.importance * float(
            first_curve.marginal_utility(np.array([first_value]))[0]
        )
        second_marginal = second.importance * float(
            second_curve.marginal_utility(np.array([second_value]))[0]
        )
        if np.isclose(second_marginal, 0.0):
            raise ZeroDivisionError("second add-on has zero marginal utility")
        return -first_marginal / second_marginal
