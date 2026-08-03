#!/usr/bin/env python
"""Compare legacy and fitted return-model moments and warmed runtime."""

from __future__ import annotations

import argparse
import gc
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median
from time import perf_counter

import numpy as np

from lifecycle_finance import Person, RegimeSwitchingMarket, StochasticMarket
from lifecycle_finance.return_models import MarketPathModel


@dataclass(frozen=True, slots=True)
class ReturnMoments:
    equity_mean: float
    bond_mean: float
    equity_standard_deviation: float
    bond_standard_deviation: float
    equity_bond_correlation: float


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _moments(model: MarketPathModel, *, paths: int, seed: int) -> ReturnMoments:
    generated = model.generate(paths=paths, horizon=1, seed=seed)
    equity_excess = generated.equity_returns[:, 0] - generated.cash_returns[:, 0]
    bond_excess = generated.bond_returns[:, 0] - generated.cash_returns[:, 0]
    return ReturnMoments(
        equity_mean=float(np.mean(equity_excess)),
        bond_mean=float(np.mean(bond_excess)),
        equity_standard_deviation=float(np.std(equity_excess)),
        bond_standard_deviation=float(np.std(bond_excess)),
        equity_bond_correlation=float(np.corrcoef(equity_excess, bond_excess)[0, 1]),
    )


def _timed_generation(
    model: MarketPathModel,
    *,
    paths: int,
    horizon: int,
    seed: int,
) -> float:
    started = perf_counter()
    generated = model.generate(paths=paths, horizon=horizon, seed=seed)
    elapsed = perf_counter() - started
    del generated
    return elapsed


def _print_moment_comparison(
    legacy: ReturnMoments,
    regime: ReturnMoments,
) -> None:
    print("Moment comparison")
    print("metric,legacy,regime")
    print(
        "equity_excess_mean,"
        f"{legacy.equity_mean:.12f},{regime.equity_mean:.12f}"
    )
    print(
        "bond_excess_mean,"
        f"{legacy.bond_mean:.12f},{regime.bond_mean:.12f}"
    )
    print(
        "equity_excess_standard_deviation,"
        f"{legacy.equity_standard_deviation:.12f},"
        f"{regime.equity_standard_deviation:.12f}"
    )
    print(
        "bond_excess_standard_deviation,"
        f"{legacy.bond_standard_deviation:.12f},"
        f"{regime.bond_standard_deviation:.12f}"
    )
    print(
        "equity_bond_correlation,"
        f"{legacy.equity_bond_correlation:.12f},"
        f"{regime.equity_bond_correlation:.12f}"
    )


def _run_performance_benchmark(
    *,
    benchmark_paths: Sequence[int],
    repeats: int,
    horizon: int,
    seed: int,
) -> None:
    print()
    print("Performance benchmark")
    print("paths,legacy_runs_seconds,legacy_median,regime_runs_seconds,regime_median,ratio")
    for paths in benchmark_paths:
        legacy = StochasticMarket()
        regime = RegimeSwitchingMarket()
        legacy.generate(paths=paths, horizon=horizon, seed=seed)
        regime.generate(paths=paths, horizon=horizon, seed=seed)

        legacy_runs: list[float] = []
        regime_runs: list[float] = []
        for _ in range(repeats):
            gc.collect()
            legacy_runs.append(
                _timed_generation(
                    legacy,
                    paths=paths,
                    horizon=horizon,
                    seed=seed,
                )
            )
            gc.collect()
            regime_runs.append(
                _timed_generation(
                    regime,
                    paths=paths,
                    horizon=horizon,
                    seed=seed,
                )
            )

        legacy_median = median(legacy_runs)
        regime_median = median(regime_runs)
        formatted_legacy_runs = "|".join(f"{run:.6f}" for run in legacy_runs)
        formatted_regime_runs = "|".join(f"{run:.6f}" for run in regime_runs)
        print(
            f"{paths},{formatted_legacy_runs},{legacy_median:.6f},"
            f"{formatted_regime_runs},{regime_median:.6f},"
            f"{regime_median / legacy_median:.6f}"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moment-paths", type=_positive_integer, default=1_000_000)
    parser.add_argument("--moment-warmup-paths", type=_positive_integer, default=1_000)
    parser.add_argument(
        "--benchmark-paths",
        type=_positive_integer,
        nargs="+",
        default=[10_000, 100_000],
    )
    parser.add_argument("--benchmark-repeats", type=_positive_integer, default=3)
    parser.add_argument("--horizon", type=_positive_integer, default=Person().horizon)
    parser.add_argument("--seed", type=int, default=20_260_724)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    legacy = StochasticMarket()
    regime = RegimeSwitchingMarket()

    legacy.generate(
        paths=args.moment_warmup_paths,
        horizon=1,
        seed=args.seed,
    )
    regime.generate(
        paths=args.moment_warmup_paths,
        horizon=1,
        seed=args.seed,
    )
    _print_moment_comparison(
        _moments(legacy, paths=args.moment_paths, seed=args.seed),
        _moments(regime, paths=args.moment_paths, seed=args.seed),
    )
    _run_performance_benchmark(
        benchmark_paths=args.benchmark_paths,
        repeats=args.benchmark_repeats,
        horizon=args.horizon,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
