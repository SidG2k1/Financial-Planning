"""Optional plotting adapters with no matplotlib import at package import time."""

from __future__ import annotations

from typing import Any

import numpy as np

from .domain import LifecyclePlan, OutcomeType
from .simulation import SimulationResult
from .sweeps import Sweep2DResult, SweepResult
from .utility import UtilityAddon


def _plt() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Install the plot extra: uv sync --extra plot") from error
    return plt


def plot_lifecycle_plan(plan: LifecyclePlan) -> Any:
    plt = _plt()
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ages = np.asarray(plan.ages)

    axes[0, 0].plot(ages, plan.human_capital_path, label="Human capital")
    axes[0, 0].plot(ages, plan.liability_path, label="Liabilities")
    axes[0, 0].set_title("Economic balance sheet")
    axes[0, 0].set_ylabel("Real dollars")
    axes[0, 0].legend()

    axes[0, 1].plot(ages, plan.income_path, label="Income")
    axes[0, 1].plot(
        ages,
        np.asarray(plan.discretionary_consumption_path)
        + float(plan.diagnostics.get("nondiscretionary_consumption", 0.0)),
        label="Consumption",
    )
    axes[0, 1].set_title("Income and consumption")
    axes[0, 1].legend()

    domestic = [allocation.domestic_equity for allocation in plan.glide_path]
    global_equity = [allocation.global_equity for allocation in plan.glide_path]
    bonds = [allocation.bonds for allocation in plan.glide_path]
    cash = [allocation.cash for allocation in plan.glide_path]
    axes[1, 0].stackplot(
        ages,
        domestic,
        global_equity,
        bonds,
        cash,
        labels=("Domestic equity", "Global equity", "Bonds", "Cash"),
    )
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_title("Constrained financial glide path")
    axes[1, 0].legend(loc="upper right", fontsize="small")

    axes[1, 1].plot(ages, plan.survival_probabilities)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title("Conditional survival probability")
    for axis in axes.flat:
        axis.set_xlabel("Age")
        axis.grid(alpha=0.2)
    return figure


def plot_simulation(result: SimulationResult) -> Any:
    plt = _plt()
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    ages = np.asarray(result.ages)
    alive = ages[np.newaxis, :] < result.death_ages[:, np.newaxis]
    has_survivors = np.any(alive, axis=0)
    wealth = np.where(alive, result.wealth_paths[:, :-1], np.nan)
    spending = np.where(alive, result.spending_paths, np.nan)
    for quantile, label in ((0.05, "5th"), (0.5, "Median"), (0.95, "95th")):
        wealth_quantile = np.full(ages.shape, np.nan)
        spending_quantile = np.full(ages.shape, np.nan)
        wealth_quantile[has_survivors] = np.nanquantile(wealth[:, has_survivors], quantile, axis=0)
        spending_quantile[has_survivors] = np.nanquantile(
            spending[:, has_survivors], quantile, axis=0
        )
        axes[0].plot(ages, wealth_quantile, label=label)
        axes[1].plot(ages, spending_quantile, label=label)
    axes[0].set_title("Financial wealth among survivors")
    axes[1].set_title("Consumption among survivors")
    for axis in axes:
        axis.set_xlabel("Age")
        axis.set_ylabel("Real dollars")
        axis.grid(alpha=0.2)
        axis.legend()
    return figure


def plot_sweep(result: SweepResult | Sweep2DResult) -> Any:
    plt = _plt()
    if isinstance(result, SweepResult):
        figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
        axis.plot(result.values, result.metrics, marker="o")
        optimum_index = result.optimum_index
        axis.scatter(
            result.values[optimum_index],
            result.metrics[optimum_index],
            color="tab:red",
            zorder=3,
            label="Utility optimum",
        )
        axis.set_xlabel(result.parameter)
        axis.set_ylabel(result.metric.replace("_", " ").title())
        axis.grid(alpha=0.2)
        axis.legend()
        return figure

    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    contour = axis.contourf(
        result.x_values,
        result.y_values,
        result.metrics,
        levels=20,
    )
    x, y, _ = result.optimum
    axis.scatter(x, y, color="tab:red", marker="*", s=120, label="Utility optimum")
    axis.set_xlabel(result.x_parameter)
    axis.set_ylabel(result.y_parameter)
    axis.legend()
    figure.colorbar(contour, ax=axis, label=result.metric.replace("_", " ").title())
    return figure


def plot_utility_curve(addon: UtilityAddon, values: Any) -> Any:
    """Render an add-on's weighted utility curve for preference inspection."""
    plt = _plt()
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 2:
        raise ValueError("values must be a one-dimensional grid with at least two points")
    y = addon.importance * addon.curve.evaluate(x)
    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    axis.plot(x, y)
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axis.set_title(addon.name.replace("_", " ").title())
    axis.set_xlabel(addon.outcome.replace("_", " ").title())
    axis.set_ylabel("Utility points")
    axis.grid(alpha=0.2)
    return figure


def plot_rolling_decisions(result: SimulationResult) -> Any:
    """Render survivor-conditional rolling controls."""
    plt = _plt()
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    ages = np.asarray(result.ages)
    alive = ages[np.newaxis, :] < result.death_ages[:, np.newaxis]
    controls = (
        (OutcomeType.ALLOCATION_EQUITY, "Equity fraction"),
        (OutcomeType.LEVERAGE, "Leverage"),
        (OutcomeType.INSURED_BEQUEST, "Insured bequest"),
    )
    for axis, (outcome, title) in zip(axes, controls, strict=True):
        values = np.where(alive, result.decision_paths[outcome], np.nan)
        has_survivors = np.any(alive, axis=0)
        median = np.full(len(ages), np.nan)
        median[has_survivors] = np.nanmedian(
            values[:, has_survivors],
            axis=0,
        )
        axis.plot(ages, median)
        axis.set_title(title)
        axis.set_xlabel("Age")
        axis.grid(alpha=0.2)
    axes[0].set_ylim(0, 1)
    return figure
