# Utility Model: Scoring lifetime spending streams
# All monetary values are in $1,000 units (2023 USD)

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from config import SimulationConfig
from models import vitality_at_age


class UtilityScorer(ABC):
    """Scores a lifetime spending stream."""

    @abstractmethod
    def score(self, spending: List[float], config: SimulationConfig) -> float:
        """Compute lifetime utility of a spending stream."""
        ...

    @abstractmethod
    def certainty_equivalent(self, utility: float, n_years: int) -> float:
        """Constant annual spending giving the same lifetime utility."""
        ...


class CRRAUtility(UtilityScorer):
    """CRRA power utility with FIRE multiplier and vitality weighting.

    V = sum beta^t * vitality(age_t) * fire_mult_t * c_t^alpha

    Vitality captures declining health/energy with age (QALY-inspired).
    FIRE multiplier captures extra value of freedom when retired.
    """

    def __init__(self, power: float = 0.8, discount_rate: float = 0.03,
                 fire_multiplier: float = 1.0,
                 retirement_year_idx: int | None = None,
                 start_age: int = 25,
                 config: SimulationConfig | None = None) -> None:
        self.power = power
        self.discount_rate = discount_rate
        self.fire_multiplier = fire_multiplier
        self.retirement_year_idx = retirement_year_idx
        self.start_age = start_age
        self._config = config or SimulationConfig()

    def _weight(self, t: int) -> float:
        """Compute weight for year t: beta^t * vitality * fire_mult."""
        beta = 1.0 / (1.0 + self.discount_rate)
        age = self.start_age + t
        v = vitality_at_age(age, self._config)
        if (self.retirement_year_idx is not None
                and t > self.retirement_year_idx):
            m = self.fire_multiplier
        else:
            m = 1.0
        return (beta ** t) * v * m

    def score(self, spending: List[float], config: SimulationConfig) -> float:
        total = 0.0
        for t, c in enumerate(spending):
            if c > 0:
                total += self._weight(t) * (c ** self.power)
        return total

    def certainty_equivalent(self, utility: float, n_years: int) -> float:
        """CE: constant spending giving same lifetime utility.

        Solves: sum w_t * CE^alpha = V  ->  CE = (V / sum w_t)^(1/alpha)
        where w_t = beta^t * vitality(age_t) * fire_mult_t.
        """
        w_sum = sum(self._weight(t) for t in range(n_years))
        if w_sum <= 0 or utility <= 0:
            return 0.0
        return (utility / w_sum) ** (1.0 / self.power)

    @classmethod
    def from_config(cls, config: SimulationConfig) -> CRRAUtility:
        """Build from SimulationConfig fields."""
        return cls(
            power=config.utility_power,
            discount_rate=config.discount_rate,
            fire_multiplier=config.fire_multiplier,
            retirement_year_idx=config.retirement_age - config.start_age,
            start_age=config.start_age,
            config=config,
        )
