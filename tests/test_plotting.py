from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from lifecycle_finance import (
    LifecyclePlanner,
    MonteCarloEngine,
    PlanningScenario,
    SimulationSettings,
    Sweep2DResult,
    SweepResult,
    UtilityAddon,
)
from lifecycle_finance.plotting import (
    plot_lifecycle_plan,
    plot_rolling_decisions,
    plot_simulation,
    plot_sweep,
    plot_utility_curve,
)
from lifecycle_finance.utility import SpendingFloorCurve


def test_all_plotting_adapters_render() -> None:
    scenario = PlanningScenario()
    plan = LifecyclePlanner().plan(scenario)
    simulation = MonteCarloEngine().simulate(
        scenario,
        settings=SimulationSettings(paths=64, seed=9),
    )
    sweep = SweepResult(
        parameter="retirement_age",
        values=(62.0, 65.0, 67.0),
        metrics=(1.0, 2.0, 1.5),
        utilities=(0.0, 1.0, 0.5),
        metric="median_certainty_equivalent",
    )
    sweep_2d = Sweep2DResult(
        x_parameter="retirement_age",
        y_parameter="leverage",
        x_values=(62.0, 65.0, 67.0),
        y_values=(1.0, 1.25, 1.5),
        metrics=np.array(
            [
                [1.0, 1.5, 1.2],
                [1.4, 2.0, 1.7],
                [1.1, 1.6, 1.3],
            ]
        ),
        utilities=np.array(
            [
                [0.0, 0.5, 0.2],
                [0.4, 1.0, 0.7],
                [0.1, 0.6, 0.3],
            ]
        ),
        metric="median_certainty_equivalent",
    )

    figures = [
        plot_lifecycle_plan(plan),
        plot_simulation(simulation),
        plot_sweep(sweep),
        plot_sweep(sweep_2d),
        plot_utility_curve(
            UtilityAddon(
                name="basic_spending",
                outcome="spending",
                curve=SpendingFloorCurve(40_000, 10_000),
                importance=2,
            ),
            np.linspace(0, 80_000, 100),
        ),
        plot_rolling_decisions(simulation),
    ]

    assert all(figure.axes for figure in figures)
    assert figures[1].axes[0].get_title() == "Financial wealth among survivors"
    plt.close("all")
