"""Command-line adapter for deterministic plans, simulation, and sweeps."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from .domain import LeverageInstrument, PlanningScenario, SimulationSettings
from .lifecycle import LifecyclePlanner
from .return_models import RegimeSwitchingMarket, StochasticMarket
from .serialization import (
    load_scenario,
    save_plan,
    save_simulation,
    save_simulation_summary,
)
from .simulation import MonteCarloEngine, SimulationResult, SimulationSummary
from .sweeps import SUPPORTED_METRICS, parameter_sweep


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _scenario(path: str | None) -> PlanningScenario:
    return PlanningScenario() if path is None else load_scenario(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lifecycle-finance",
        description="Lifecycle planning and stochastic retirement simulation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("example", help="print the default workbook-style scenario")

    plan = subparsers.add_parser("plan", help="run the deterministic lifecycle planner")
    plan.add_argument("--scenario", help="scenario JSON file")
    plan.add_argument("--output", help="write the full plan to JSON")

    simulate = subparsers.add_parser("simulate", help="run vectorized Monte Carlo")
    simulate.add_argument("--scenario", help="scenario JSON file")
    simulate.add_argument("--paths", type=int, default=1_000)
    simulate.add_argument("--seed", type=int, default=42)
    simulate.add_argument(
        "--market-model",
        choices=("regime", "legacy"),
        default="regime",
    )
    simulate.add_argument("--leverage", type=float, default=1.0)
    simulate.add_argument(
        "--instrument",
        choices=("generic", "futures", "box_spread"),
        default="generic",
    )
    simulate.add_argument("--output", help="write complete paths to a compressed NPZ")
    simulate.add_argument(
        "--chunk-size",
        type=int,
        help="bound temporary memory by processing this many paths at a time",
    )
    simulate.add_argument(
        "--summary-only",
        action="store_true",
        help="retain scalar outcomes instead of complete annual paths",
    )
    simulate.add_argument("--deterministic-lifespan", action="store_true")
    simulate.add_argument("--deterministic-income", action="store_true")

    sweep = subparsers.add_parser(
        "sweep",
        help="run a common-random-number, utility-ranked 1D decision sweep",
    )
    sweep.add_argument("parameter")
    sweep.add_argument("values", nargs="+", type=float)
    sweep.add_argument("--scenario", help="scenario JSON file")
    sweep.add_argument("--paths", type=int, default=500)
    sweep.add_argument("--seed", type=int, default=42)
    sweep.add_argument(
        "--market-model",
        choices=("regime", "legacy"),
        default="regime",
    )
    sweep.add_argument(
        "--metric",
        choices=SUPPORTED_METRICS,
        default="mean_utility",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "example":
        print(
            json.dumps(
                asdict(PlanningScenario()),
                indent=2,
                default=_json_default,
            )
        )
        return 0

    scenario = _scenario(args.scenario)
    if args.command == "plan":
        plan_result = LifecyclePlanner().plan(scenario)
        if args.output:
            save_plan(plan_result, args.output)
        summary = {
            "financial_wealth": plan_result.financial_wealth,
            "human_capital": plan_result.human_capital,
            "liabilities": (
                plan_result.consumption_liability + plan_result.life_insurance_liability
            ),
            "net_worth": plan_result.net_worth,
            "initial_discretionary_consumption": (plan_result.initial_discretionary_consumption),
            "bequest": plan_result.bequest,
            "constrained_allocation": plan_result.constrained_allocation.as_dict(),
        }
        print(json.dumps(summary, indent=2, default=_json_default))
        return 0

    settings = SimulationSettings(
        paths=args.paths,
        seed=args.seed,
        leverage=getattr(args, "leverage", 1.0),
        leverage_instrument=LeverageInstrument(getattr(args, "instrument", "generic")),
        stochastic_lifespan=not getattr(args, "deterministic_lifespan", False),
        stochastic_income=not getattr(args, "deterministic_income", False),
    )
    market = RegimeSwitchingMarket() if args.market_model == "regime" else StochasticMarket()
    engine = MonteCarloEngine(market=market)
    if args.command == "simulate":
        simulation_result: SimulationResult | SimulationSummary
        if args.chunk_size is not None or args.summary_only:
            chunk_size = (
                args.chunk_size
                if args.chunk_size is not None
                else min(args.paths, 10_000)
            )
            if args.output and not args.summary_only:
                simulation_result = engine.simulate_chunked(
                    scenario,
                    settings=settings,
                    chunk_size=chunk_size,
                    retain_paths=True,
                )
            else:
                simulation_result = engine.simulate_chunked(
                    scenario,
                    settings=settings,
                    chunk_size=chunk_size,
                    retain_paths=False,
                )
        else:
            simulation_result = engine.simulate(scenario, settings=settings)
        if args.output:
            destination = Path(args.output)
            if isinstance(simulation_result, SimulationResult):
                save_simulation(simulation_result, destination)
            else:
                save_simulation_summary(simulation_result, destination)
        print(
            json.dumps(
                simulation_result.summary(),
                indent=2,
                default=_json_default,
            )
        )
        return 0

    if args.command == "sweep":
        sweep_result = parameter_sweep(
            engine,
            scenario,
            settings,
            parameter=args.parameter,
            values=args.values,
            metric=args.metric,
        )
        payload = asdict(sweep_result)
        payload["optimum"] = sweep_result.optimum
        print(json.dumps(payload, indent=2, default=_json_default))
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
