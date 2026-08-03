"""Stable JSON and compressed-array serialization."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .domain import (
    EconomicExposure,
    FinancialWealth,
    IncomePlan,
    Person,
    PlanningScenario,
    Preferences,
    UtilityAddonConfig,
)
from .simulation import SimulationResult, SimulationSummary


def scenario_to_dict(scenario: PlanningScenario) -> dict[str, Any]:
    return asdict(scenario)


def scenario_from_dict(data: Mapping[str, Any]) -> PlanningScenario:
    allowed = {
        "person",
        "wealth",
        "income",
        "preferences",
        "human_capital_exposure",
        "liability_exposure",
        "utility_addons",
    }
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"unknown scenario fields: {sorted(unknown)}")
    return PlanningScenario(
        person=Person(**data.get("person", {})),
        wealth=FinancialWealth(**data.get("wealth", {})),
        income=IncomePlan(**data.get("income", {})),
        preferences=Preferences(**data.get("preferences", {})),
        human_capital_exposure=EconomicExposure(
            **data.get(
                "human_capital_exposure",
                {"equity_fraction": 0.20, "global_fraction_of_equity": 0.25},
            )
        ),
        liability_exposure=EconomicExposure(
            **data.get(
                "liability_exposure",
                {"equity_fraction": 0.15, "global_fraction_of_equity": 0.0},
            )
        ),
        utility_addons=tuple(
            UtilityAddonConfig(**config) for config in data.get("utility_addons", [])
        ),
    )


def load_scenario(path: str | Path) -> PlanningScenario:
    source = Path(path)
    return scenario_from_dict(json.loads(source.read_text()))


def save_scenario(scenario: PlanningScenario, path: str | Path) -> None:
    destination = Path(path)
    destination.write_text(json.dumps(scenario_to_dict(scenario), indent=2) + "\n")


def save_plan(plan: Any, path: str | Path) -> None:
    destination = Path(path)
    payload = plan.to_dict() if hasattr(plan, "to_dict") else asdict(plan)
    destination.write_text(json.dumps(payload, indent=2) + "\n")


def save_simulation(result: SimulationResult, path: str | Path) -> None:
    """Save full paths in compressed NPZ form with a JSON metadata scalar."""
    destination = Path(path)
    metadata = {
        "ages": result.ages,
        "seed": result.seed,
        "summary": result.summary(),
        "lifecycle_plan": result.lifecycle_plan.to_dict(),
    }
    arrays: dict[str, Any] = {
        "wealth_paths": result.wealth_paths,
        "spending_paths": result.spending_paths,
        "income_paths": result.income_paths,
        "death_ages": result.death_ages,
        "insolvent": result.diagnostics.insolvent,
        "policy_shortfall": result.diagnostics.policy_shortfall,
        "margin_calls": result.margin_calls,
        "certainty_equivalents": result.certainty_equivalents,
        "utility_scores": result.utility_scores,
        "metadata": np.array(json.dumps(metadata)),
    }
    arrays.update(
        {f"utility__{name}": values for name, values in result.utility_component_scores.items()}
    )
    arrays.update(
        {f"decision__{name.value}": values for name, values in result.decision_paths.items()}
    )
    for name, diagnostic in result.diagnostics.preferences.items():
        arrays[f"preference_breach__{name}"] = diagnostic.breach_count
        arrays[f"preference_loss__{name}"] = diagnostic.utility_loss
    with destination.open("wb") as archive:
        np.savez_compressed(archive, **arrays)


def save_simulation_summary(
    result: SimulationSummary,
    path: str | Path,
) -> None:
    """Save bounded-memory scalar path outcomes and metadata."""
    destination = Path(path)
    metadata = {
        "paths": result.paths,
        "seed": result.seed,
        "summary": result.summary(),
        "lifecycle_plan": result.lifecycle_plan.to_dict(),
    }
    arrays: dict[str, Any] = {
        "terminal_wealth": result.terminal_wealth,
        "certainty_equivalents": result.certainty_equivalents,
        "utility_scores": result.utility_scores,
        "insolvent": result.diagnostics.insolvent,
        "policy_shortfall": result.diagnostics.policy_shortfall,
        "margin_calls": result.margin_calls,
        "metadata": np.array(json.dumps(metadata)),
    }
    arrays.update(
        {f"utility__{name}": values for name, values in result.utility_component_scores.items()}
    )
    for name, diagnostic in result.diagnostics.preferences.items():
        arrays[f"preference_breach__{name}"] = diagnostic.breach_count
        arrays[f"preference_loss__{name}"] = diagnostic.utility_loss
    with destination.open("wb") as archive:
        np.savez_compressed(archive, **arrays)
