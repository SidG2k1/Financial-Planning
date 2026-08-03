#!/usr/bin/env python
"""Fit and emit the default three-state return-model calibration."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from lifecycle_finance.regime_calibration import (
    HistoricalReturns,
    file_sha256,
    fit_regime_model,
    render_defaults_module,
)

DEFAULT_SOURCE_URL = (
    "https://pages.stern.nyu.edu/~adamodar/pc/datasets/histretSP.xls"
)
DEFAULT_OUTPUT = (
    Path(__file__).parents[1]
    / "src"
    / "lifecycle_finance"
    / "calibrated_regime_defaults.py"
)


def _load_csv(path: Path) -> HistoricalReturns:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "year",
        "real_equity_return",
        "real_cash_return",
        "real_bond_return",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(
            "CSV requires year, real_equity_return, real_cash_return, "
            "and real_bond_return columns"
        )
    return HistoricalReturns(
        years=np.array([int(row["year"]) for row in rows], dtype=np.int64),
        real_equity_returns=np.array(
            [float(row["real_equity_return"]) for row in rows]
        ),
        real_cash_returns=np.array([float(row["real_cash_return"]) for row in rows]),
        real_bond_returns=np.array([float(row["real_bond_return"]) for row in rows]),
    )


def _load_damodaran_workbook(path: Path) -> HistoricalReturns:
    try:
        import pandas as pd
    except ImportError as error:
        raise RuntimeError(
            "workbook calibration requires the calibration extra: "
            "uv run --extra calibration python tools/calibrate_regime_model.py ..."
        ) from error

    raw = pd.read_excel(path, sheet_name="Returns by year", header=None)
    header_rows = raw.index[raw.iloc[:, 0].eq("Year")].tolist()
    if len(header_rows) != 1:
        raise ValueError("could not locate the Returns by year workbook header")
    header_index = int(header_rows[0])
    header = raw.iloc[header_index].astype(str)
    expected = {
        21: "S&P 500 (includes dividends)2",
        23: "3-month T. Bill (Real)",
        24: "!0-year T.Bonds",
    }
    for column, name in expected.items():
        if header.iloc[column] != name:
            raise ValueError(
                f"unexpected workbook column {column}: {header.iloc[column]!r}"
            )

    values = raw.iloc[header_index + 1 :, [0, 21, 23, 24]].copy()
    values.iloc[:, 0] = pd.to_numeric(values.iloc[:, 0], errors="coerce")
    values = values.loc[values.iloc[:, 0].notna()].astype(float)
    return HistoricalReturns(
        years=values.iloc[:, 0].to_numpy(dtype=np.int64),
        real_equity_returns=values.iloc[:, 1].to_numpy(dtype=float),
        real_cash_returns=values.iloc[:, 2].to_numpy(dtype=float),
        real_bond_returns=values.iloc[:, 3].to_numpy(dtype=float),
    )


def load_history(path: Path) -> HistoricalReturns:
    if path.suffix.lower() == ".csv":
        return _load_csv(path)
    if path.suffix.lower() == ".xls":
        return _load_damodaran_workbook(path)
    raise ValueError("input must be a .csv or .xls file")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the existing output differs instead of rewriting it",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    history = load_history(args.input)
    fitted = fit_regime_model(history)
    rendered = render_defaults_module(
        fitted,
        source_url=args.source_url,
        source_sha256=file_sha256(args.input),
        first_year=int(history.years[0]),
        last_year=int(history.years[-1]),
    )
    if args.check:
        if not args.output.exists() or args.output.read_text() != rendered:
            print(f"calibration output is stale: {args.output}")
            return 1
        print(f"calibration output is current: {args.output}")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
