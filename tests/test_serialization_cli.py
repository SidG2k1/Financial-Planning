import json

import numpy as np
import pytest

import lifecycle_finance.cli as cli_module
from lifecycle_finance import (
    MonteCarloEngine,
    PlanningScenario,
    RegimeSwitchingMarket,
    SimulationSettings,
    StaticPriceProvider,
    StochasticMarket,
    UtilityAddonConfig,
    UtilityAggregation,
    UtilityCurveKind,
)
from lifecycle_finance.cli import build_parser, main
from lifecycle_finance.serialization import (
    load_scenario,
    save_plan,
    save_scenario,
    save_simulation,
    save_simulation_summary,
)
from lifecycle_finance.sweeps import SUPPORTED_METRICS


def test_sweep_metrics_are_shared_with_the_cli() -> None:
    expected = (
        "mean_utility",
        "median_utility",
        "median_certainty_equivalent",
        "mean_certainty_equivalent",
        "insolvency_probability",
        "policy_shortfall_probability",
        "median_terminal_wealth",
    )
    sweep_parser = next(
        action
        for action in build_parser()._subparsers._group_actions[0].choices["sweep"]._actions
        if action.dest == "metric"
    )

    assert expected == SUPPORTED_METRICS
    assert sweep_parser.choices == SUPPORTED_METRICS


@pytest.mark.parametrize(
    ("command", "model_name", "model_type"),
    [
        ("simulate", "regime", RegimeSwitchingMarket),
        ("simulate", "legacy", StochasticMarket),
        ("sweep", "regime", RegimeSwitchingMarket),
        ("sweep", "legacy", StochasticMarket),
    ],
)
def test_cli_selects_concrete_market_model_for_simulation_and_sweep(
    command: str,
    model_name: str,
    model_type: type[RegimeSwitchingMarket] | type[StochasticMarket],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    selected_markets: list[RegimeSwitchingMarket | StochasticMarket] = []

    def recording_engine(*, market: RegimeSwitchingMarket | StochasticMarket) -> MonteCarloEngine:
        selected_markets.append(market)
        return MonteCarloEngine(market=market)

    monkeypatch.setattr(cli_module, "MonteCarloEngine", recording_engine)
    argv = (
        [command, "--paths", "2", "--market-model", model_name]
        if command == "simulate"
        else [command, "leverage", "1.0", "--paths", "2", "--market-model", model_name]
    )

    assert main(argv) == 0
    assert isinstance(selected_markets[0], model_type)
    assert json.loads(capsys.readouterr().out)


def test_scenario_json_round_trip(tmp_path) -> None:
    path = tmp_path / "scenario.json"
    scenario = PlanningScenario(
        utility_addons=(
            UtilityAddonConfig(
                name="retirement_freedom",
                outcome="retired",
                curve=UtilityCurveKind.LINEAR,
                parameters={"slope": 1},
                importance=3,
                aggregation=UtilityAggregation.DISCOUNTED_SUM,
                age_reference=40,
                age_growth=0.02,
            ),
        )
    )
    save_scenario(scenario, path)
    loaded = load_scenario(path)
    assert loaded == scenario


def test_static_price_provider_is_offline_and_validated() -> None:
    provider = StaticPriceProvider({"A": 10.0, "B": 20.0})
    assert provider.prices(("B", "A")) == {"B": 20.0, "A": 10.0}


def test_plan_and_simulation_serialization(tmp_path) -> None:
    result = MonteCarloEngine().simulate(settings=SimulationSettings(paths=4, seed=21))
    plan_path = tmp_path / "plan.json"
    simulation_path = tmp_path / "simulation.npz"
    save_plan(result.lifecycle_plan, plan_path)
    save_simulation(result, simulation_path)

    plan = json.loads(plan_path.read_text())
    assert plan["financial_wealth"] == result.lifecycle_plan.financial_wealth
    archive = np.load(simulation_path)
    assert archive["wealth_paths"].shape == result.wealth_paths.shape
    assert "insolvent" in archive
    assert "policy_shortfall" in archive
    assert "decision__leverage" in archive
    metadata = json.loads(str(archive["metadata"]))
    assert metadata["seed"] == 21


def test_save_simulation_respects_path_without_npz_suffix(tmp_path) -> None:
    result = MonteCarloEngine().simulate(settings=SimulationSettings(paths=4, seed=21))
    destination = tmp_path / "simulation"

    save_simulation(result, destination)

    assert destination.is_file()
    assert not destination.with_suffix(".npz").exists()
    with np.load(destination) as archive:
        assert archive["wealth_paths"].shape == result.wealth_paths.shape


def test_save_simulation_summary_respects_path_without_npz_suffix(tmp_path) -> None:
    result = MonteCarloEngine().simulate_chunked(
        settings=SimulationSettings(paths=4, seed=21),
        chunk_size=2,
        retain_paths=False,
    )
    destination = tmp_path / "summary"

    save_simulation_summary(result, destination)

    assert destination.is_file()
    assert not destination.with_suffix(".npz").exists()
    with np.load(destination) as archive:
        assert archive["terminal_wealth"].shape == (4,)


def test_cli_plan_and_simulate(tmp_path, capsys) -> None:
    assert main(["plan"]) == 0
    plan_output = json.loads(capsys.readouterr().out)
    assert plan_output["net_worth"] > 0

    output = tmp_path / "paths.npz"
    assert (
        main(
            [
                "simulate",
                "--paths",
                "4",
                "--seed",
                "3",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    simulation_output = json.loads(capsys.readouterr().out)
    assert simulation_output["paths"] == 4
    assert output.exists()

    summary_output = tmp_path / "summary.npz"
    assert (
        main(
            [
                "simulate",
                "--paths",
                "9",
                "--seed",
                "3",
                "--chunk-size",
                "4",
                "--summary-only",
                "--output",
                str(summary_output),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["paths"] == 9
    archive = np.load(summary_output)
    assert archive["terminal_wealth"].shape == (9,)
    assert "wealth_paths" not in archive
