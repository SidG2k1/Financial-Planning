#!/usr/bin/env python
"""Benchmark end-to-end lifecycle simulation time and summary peak memory."""

from __future__ import annotations

import argparse
import gc
import json
import resource
import shlex
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

import numpy as np

REPOSITORY_SOURCE = str(Path(__file__).resolve().parents[1] / "src")
if REPOSITORY_SOURCE in sys.path:
    sys.path.remove(REPOSITORY_SOURCE)
sys.path.insert(0, REPOSITORY_SOURCE)

from lifecycle_finance import (  # noqa: E402
    DerivativeSpendingSolver,
    MonteCarloEngine,
    PlanningScenario,
    SimulationSettings,
)
from lifecycle_finance.spending import (  # noqa: E402
    SpendingOptimizationProblem,
    build_spending_problem,
)

DEFAULT_SEED = 20_260_726


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=_positive_integer, default=10_000)
    parser.add_argument("--repeats", type=_positive_integer, default=3)
    parser.add_argument("--chunk-size", type=_positive_integer, default=10_000)
    parser.add_argument("--seed", type=_nonnegative_integer, default=DEFAULT_SEED)
    parser.add_argument("--memory-only", action="store_true")
    parser.add_argument("--rss-child", action="store_true", help=argparse.SUPPRESS)
    return parser


def _maximum_rss_bytes() -> int:
    maximum = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return maximum * 1024 if sys.platform.startswith("linux") else maximum


def _settings(args: argparse.Namespace) -> SimulationSettings:
    return SimulationSettings(paths=args.paths, seed=args.seed)


def _solver_problem(
    engine: MonteCarloEngine,
    scenario: PlanningScenario,
    *,
    paths: int,
) -> SpendingOptimizationProblem:
    prepared = engine._prepare(scenario)
    survival = np.asarray(prepared.plan.survival_probabilities, dtype=float)
    conditional_survival = survival / survival[0]
    discount = 1.0 / (1.0 + engine.planner.capital_markets.risk_free_rate)
    offsets = np.arange(len(conditional_survival), dtype=float)
    future_annuity = float(
        np.sum(conditional_survival[1:] * discount ** offsets[1:])
    )
    resources = np.linspace(
        prepared.plan.financial_wealth,
        prepared.plan.net_worth,
        paths,
        dtype=float,
    )
    return build_spending_problem(
        utility_model=prepared.utility_model,
        resources=resources,
        future_annuity=np.full(paths, future_annuity),
        ages=np.asarray(prepared.plan.ages, dtype=float),
        conditional_survival=conditional_survival,
    )


def _timed_runs(operation: Callable[[], object], *, repeats: int) -> list[float]:
    warm_result = operation()
    del warm_result
    gc.collect()
    runs: list[float] = []
    for _ in range(repeats):
        gc.collect()
        started = perf_counter()
        result = operation()
        elapsed = perf_counter() - started
        del result
        runs.append(elapsed)
    return runs


def _timing_payload(args: argparse.Namespace) -> tuple[dict[str, float], dict[str, list[float]]]:
    scenario = PlanningScenario()
    engine = MonteCarloEngine()
    settings = _settings(args)
    solver = DerivativeSpendingSolver()
    problem = _solver_problem(engine, scenario, paths=args.paths)

    operations: dict[str, Callable[[], object]] = {
        "solver_seconds": lambda: solver.solve_with_score(problem),
        "full_seconds": lambda: engine.simulate(scenario, settings=settings),
        "summary_seconds": lambda: engine.simulate_chunked(
            scenario,
            settings=settings,
            chunk_size=args.chunk_size,
            retain_paths=False,
        ),
    }
    raw_runs = {
        name: _timed_runs(operation, repeats=args.repeats)
        for name, operation in operations.items()
    }
    timings = {name: float(median(runs)) for name, runs in raw_runs.items()}
    return timings, raw_runs


def _rss_child_payload(args: argparse.Namespace) -> dict[str, int]:
    scenario = PlanningScenario()
    engine = MonteCarloEngine()
    settings = _settings(args)
    baseline = _maximum_rss_bytes()
    result = engine.simulate_chunked(
        scenario,
        settings=settings,
        chunk_size=args.chunk_size,
        retain_paths=False,
    )
    peak = _maximum_rss_bytes()
    del result
    return {
        "summary_peak_rss": peak,
        "summary_baseline_rss": baseline,
        "summary_increment_rss": max(peak - baseline, 0),
    }


def _fresh_rss_payload(args: argparse.Namespace) -> dict[str, int]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--paths",
        str(args.paths),
        "--repeats",
        str(args.repeats),
        "--chunk-size",
        str(args.chunk_size),
        "--seed",
        str(args.seed),
        "--rss-child",
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no child output"
        raise RuntimeError(
            f"RSS child failed with exit code {completed.returncode}: {detail}; "
            f"command: {shlex.join(command)}"
        )
    payload = json.loads(completed.stdout)
    return {
        name: int(payload[name])
        for name in (
            "summary_peak_rss",
            "summary_baseline_rss",
            "summary_increment_rss",
        )
    }


def _output_payload(args: argparse.Namespace) -> dict[str, Any]:
    scenario = PlanningScenario()
    if args.memory_only:
        timings: dict[str, float | None] = {
            "solver_seconds": None,
            "full_seconds": None,
            "summary_seconds": None,
        }
        raw_runs: dict[str, list[float]] = {
            name: [] for name in timings
        }
    else:
        measured, raw_runs = _timing_payload(args)
        timings = measured
    return {
        "schema_version": 1,
        "config": {
            "paths": args.paths,
            "repeats": args.repeats,
            "chunk_size": args.chunk_size,
            "seed": args.seed,
            "horizon": scenario.person.horizon,
            "memory_only": args.memory_only,
        },
        **timings,
        "raw_runs": raw_runs,
        **_fresh_rss_payload(args),
        "rss_unit": "bytes",
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.rss_child:
        print(json.dumps(_rss_child_payload(args), sort_keys=True))
        return 0
    print(json.dumps(_output_payload(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
