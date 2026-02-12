# Sweep engines: Monte Carlo, leverage sweep, retirement sweep, 2D sweep,
# instrument comparison
# All monetary values are in $1,000 units (2023 USD)

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Dict, List, Tuple

import numpy as np

from config import SimulationConfig, SimulationResult, format_money
from models import sample_bayesian_config
from simulator import run_simulation
from spending import (
    AmortizedSpending,
    FixedSpending,
    MarginalUtilitySpending,
    SpendingRule,
    VitalityAmortizedSpending,
)
from utility import CRRAUtility, UtilityScorer


# ---------------------------------------------------------------------------
# Shared helpers for CRN / Bayesian / Antithetic boilerplate
# ---------------------------------------------------------------------------

def _generate_seeds(n_simulations: int) -> List[int]:
    """CRN: deterministic seed list for cross-parameter reproducibility."""
    base_rng = np.random.default_rng(42)
    return [int(base_rng.integers(0, 1_000_000)) for _ in range(n_simulations)]


def _run_sim_batch(
    config: SimulationConfig,
    spending_rule: SpendingRule,
    seeds: List[int],
) -> List[Tuple[SimulationResult, SimulationResult | None]]:
    """Run batch of sims with CRN, bayesian sampling, and antithetic pairs."""
    batch = []
    for seed in seeds:
        sim_cfg = config
        if config.bayesian:
            sim_cfg = sample_bayesian_config(config, np.random.default_rng(seed))
        result = run_simulation(sim_cfg, spending_rule, seed=seed, quiet=True)
        result2 = None
        if config.antithetic:
            result2 = run_simulation(sim_cfg, spending_rule, seed=seed,
                                     quiet=True, negate_shocks=True)
        batch.append((result, result2))
    return batch


def _score_utility(
    scorer: UtilityScorer,
    result: SimulationResult,
    result2: SimulationResult | None,
    portfolio_idx: int,
    config: SimulationConfig,
) -> float:
    """Score one portfolio's utility, averaging antithetic pair if present."""
    u = scorer.score(result.spending[portfolio_idx], config)
    if result2 is not None:
        u2 = scorer.score(result2.spending[portfolio_idx], config)
        u = (u + u2) / 2
    return u


def evaluate_point(
    config: SimulationConfig,
    retirement_age: float,
    leverage_ratio: float,
    spending_rule: SpendingRule,
    seeds: List[int],
    portfolio_idx: int = 2,
) -> float:
    """Evaluate E[U] at a single (retirement_age, leverage) point."""
    cfg = replace(config,
                  retirement_age=int(round(retirement_age)),
                  leverage_ratio=max(leverage_ratio, 1.0))
    scorer = CRRAUtility.from_config(cfg)
    batch = _run_sim_batch(cfg, spending_rule, seeds)
    utilities = [_score_utility(scorer, result, result2, portfolio_idx, cfg)
                 for result, result2 in batch]
    return float(np.mean(utilities))


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

def run_monte_carlo(
    config: SimulationConfig,
    spending_rule: SpendingRule | None = None,
    utility_scorer: UtilityScorer | None = None,
    n_simulations: int = 500,
) -> Dict[str, Dict]:
    """Run multiple simulations and return percentile statistics + utility."""
    if spending_rule is None:
        spending_rule = FixedSpending(config.initial_expenses, config.lifestyle_inflation)
    if utility_scorer is None:
        utility_scorer = CRRAUtility.from_config(config)

    all_histories: Dict[str, List[List[float]]] = {}
    all_spending: Dict[str, List[List[float]]] = {}
    all_utilities: Dict[str, List[float]] = {}
    sim_years = config.expected_lifespan - config.start_age + 1

    seeds = _generate_seeds(n_simulations)
    batch = _run_sim_batch(config, spending_rule, seeds)

    for result, result2 in batch:
        for j, p in enumerate(result.portfolios.portfolios):
            name = type(p).__name__
            if name not in all_histories:
                all_histories[name] = []
                all_spending[name] = []
                all_utilities[name] = []
            all_histories[name].append(p.get_nw_history())
            all_spending[name].append(result.spending[j])
            u = _score_utility(utility_scorer, result, result2, j, config)

            if result2 is not None:
                all_histories[name].append(
                    result2.portfolios.portfolios[j].get_nw_history())
                all_spending[name].append(result2.spending[j])

            all_utilities[name].append(u)

    # Pad histories to common length (stochastic lifespan -> variable lengths)
    first_key = next(iter(all_histories))
    max_len = max(len(h) for h in all_histories[first_key])
    ages = list(range(config.start_age, config.start_age + max_len))

    results: Dict[str, Dict] = {}
    for name, histories in all_histories.items():
        padded = [h + [float('nan')] * (max_len - len(h)) for h in histories]
        arr = np.array(padded) / 1000
        u_arr = np.array(all_utilities[name])
        mean_u = float(np.mean(u_arr))
        median_u = float(np.median(u_arr))

        # Spending percentiles (in $1k units, no /1000 conversion)
        spend_padded = [s + [float('nan')] * (max_len - len(s))
                        for s in all_spending[name]]
        spend_arr = np.array(spend_padded)

        results[name] = {
            'ages': ages,
            'median': np.nanmedian(arr, axis=0),
            'p10': np.nanpercentile(arr, 10, axis=0),
            'p25': np.nanpercentile(arr, 25, axis=0),
            'p75': np.nanpercentile(arr, 75, axis=0),
            'p90': np.nanpercentile(arr, 90, axis=0),
            'spend_median': np.nanmedian(spend_arr, axis=0),
            'spend_p10': np.nanpercentile(spend_arr, 10, axis=0),
            'spend_p25': np.nanpercentile(spend_arr, 25, axis=0),
            'spend_p75': np.nanpercentile(spend_arr, 75, axis=0),
            'spend_p90': np.nanpercentile(spend_arr, 90, axis=0),
            'mean_utility': mean_u,
            'mean_ce': utility_scorer.certainty_equivalent(mean_u, sim_years),
            'median_utility': median_u,
            'median_ce': utility_scorer.certainty_equivalent(median_u, sim_years),
        }

    return results


# ---------------------------------------------------------------------------
# Leverage Sweep
# ---------------------------------------------------------------------------

def run_leverage_sweep(
    config: SimulationConfig,
    spending_rule: SpendingRule | None = None,
    utility_scorer: UtilityScorer | None = None,
    leverage_range: List[float] | None = None,
    n_simulations: int = 500,
) -> Dict[str, List]:
    """Sweep across leverage ratios and compute risk/reward statistics."""
    if spending_rule is None:
        spending_rule = FixedSpending(config.initial_expenses, config.lifestyle_inflation)
    if utility_scorer is None:
        utility_scorer = CRRAUtility.from_config(config)
    if leverage_range is None:
        leverage_range = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5]

    sim_years = config.expected_lifespan - config.start_age + 1

    results: Dict[str, List] = {
        'leverage': [], 'median_nw': [],
        'p10_nw': [], 'p25_nw': [], 'p75_nw': [], 'p90_nw': [],
        'ruin_pct': [], 'mean_utility': [], 'median_utility': [],
        'mean_ce': [], 'median_ce': [],
    }

    seeds = _generate_seeds(n_simulations)

    for lev in leverage_range:
        cfg = replace(config, leverage_ratio=lev)
        batch = _run_sim_batch(cfg, spending_rule, seeds)

        final_nws = []
        utilities = []
        ruin_count = 0

        for result, result2 in batch:
            lev_portfolio = result.portfolios.portfolios[2]
            history = lev_portfolio.get_nw_history()
            final_nws.append(history[-1])
            if min(history) < 0:
                ruin_count += 1
            u = _score_utility(utility_scorer, result, result2, 2, cfg)
            utilities.append(u)

        arr = np.array(final_nws) / 1000
        mean_u = float(np.mean(utilities))
        median_u = float(np.median(utilities))
        mean_ce = utility_scorer.certainty_equivalent(mean_u, sim_years)
        median_ce = utility_scorer.certainty_equivalent(median_u, sim_years)

        results['leverage'].append(lev)
        results['median_nw'].append(float(np.median(arr)))
        results['p10_nw'].append(float(np.percentile(arr, 10)))
        results['p25_nw'].append(float(np.percentile(arr, 25)))
        results['p75_nw'].append(float(np.percentile(arr, 75)))
        results['p90_nw'].append(float(np.percentile(arr, 90)))
        results['ruin_pct'].append(ruin_count / n_simulations)
        results['mean_utility'].append(mean_u)
        results['median_utility'].append(median_u)
        results['mean_ce'].append(mean_ce)
        results['median_ce'].append(median_ce)

        print(f"  {lev:.2f}x:  median=${results['median_nw'][-1]*1000:>10,.0f}k  "
              f"p10=${results['p10_nw'][-1]*1000:>10,.0f}k  "
              f"ruin={results['ruin_pct'][-1]:.1%}  "
              f"E[U]={mean_u:,.1f}  (CE={format_money(mean_ce)}/yr)")

    return results


# ---------------------------------------------------------------------------
# Retirement Age Sweep
# ---------------------------------------------------------------------------

def run_retirement_sweep(
    config: SimulationConfig,
    spending_rule: SpendingRule | None = None,
    utility_scorer_factory: Callable[[SimulationConfig], UtilityScorer] | None = None,
    retirement_ages: List[int] | None = None,
    n_simulations: int = 500,
) -> Dict[str, Dict]:
    """Sweep retirement age and find utility-optimal retirement age.

    For each candidate retirement age, runs Monte Carlo with amortized
    spending and computes E[utility] per portfolio strategy.

    Args:
        utility_scorer_factory: Takes a config (with updated retirement_age)
            and returns a UtilityScorer. Default builds CRRAUtility with
            the config's fire_multiplier.
    """
    if spending_rule is None:
        if config.vitality_floor < 1.0:
            spending_rule = MarginalUtilitySpending()
        else:
            spending_rule = AmortizedSpending()
    if utility_scorer_factory is None:
        def utility_scorer_factory(cfg: SimulationConfig) -> UtilityScorer:
            return CRRAUtility.from_config(cfg)
    if retirement_ages is None:
        retirement_ages = list(range(30, 71, 2))

    sim_years = config.expected_lifespan - config.start_age + 1
    portfolio_results: Dict[str, Dict] = {}

    seeds = _generate_seeds(n_simulations)

    for ret_age in retirement_ages:
        cfg = replace(config, retirement_age=ret_age)
        scorer = utility_scorer_factory(cfg)
        batch = _run_sim_batch(cfg, spending_rule, seeds)

        utilities_by_portfolio: Dict[str, List[float]] = {}

        for result, result2 in batch:
            for j, p in enumerate(result.portfolios.portfolios):
                name = type(p).__name__
                if name not in utilities_by_portfolio:
                    utilities_by_portfolio[name] = []
                u = _score_utility(scorer, result, result2, j, cfg)
                utilities_by_portfolio[name].append(u)

        # Aggregate
        parts = []
        for name, utils in utilities_by_portfolio.items():
            if name not in portfolio_results:
                portfolio_results[name] = {
                    'retirement_ages': [],
                    'mean_utilities': [],
                    'mean_ces': [],
                    'median_ces': [],
                }
            u_arr = np.array(utils)
            mean_u = float(np.mean(u_arr))
            median_u = float(np.median(u_arr))
            portfolio_results[name]['retirement_ages'].append(ret_age)
            portfolio_results[name]['mean_utilities'].append(mean_u)
            portfolio_results[name]['mean_ces'].append(
                scorer.certainty_equivalent(mean_u, sim_years))
            portfolio_results[name]['median_ces'].append(
                scorer.certainty_equivalent(median_u, sim_years))
            ce_str = format_money(portfolio_results[name]['mean_ces'][-1])
            parts.append(f"{name[:15]}: E[U]={mean_u:,.1f} (CE={ce_str}/yr)")

        print(f"  Retire@{ret_age}: {' | '.join(parts)}")

    # Find optimal age per portfolio
    for name, data in portfolio_results.items():
        best_idx = int(np.argmax(data['mean_ces']))
        data['optimal_age'] = data['retirement_ages'][best_idx]
        data['optimal_ce'] = data['mean_ces'][best_idx]

    return portfolio_results


# ---------------------------------------------------------------------------
# 2D Parameter Sweep
# ---------------------------------------------------------------------------

PARAM_MAP = {
    'retirement-age': 'retirement_age',
    'leverage': 'leverage_ratio',
    'erp': 'equity_risk_premium',
    'stock-vol': 'stock_vol',
    'fire-multiplier': 'fire_multiplier',
    'initial-nw': 'initial_net_worth',
    'spending-floor': 'spending_floor',
    'discount-rate': 'discount_rate',
}

PARAM_RANGES = {
    'retirement_age': (30, 70, 21),
    'leverage_ratio': (1.0, 3.5, 11),
    'equity_risk_premium': (0.02, 0.08, 13),
    'stock_vol': (0.05, 0.20, 16),
    'fire_multiplier': (1.0, 3.0, 11),
    'initial_net_worth': (100, 2000, 15),
    'spending_floor': (0, 80, 17),
    'discount_rate': (0.01, 0.06, 11),
}

LOG_SCALE_PARAMS = {'leverage_ratio', 'initial_net_worth'}

PORTFOLIO_NAME_MAP = {'cash': 0, 'stock': 1, 'leveraged': 2}


def _make_param_range(param: str) -> List[float]:
    """Generate default range for a parameter."""
    lo, hi, n = PARAM_RANGES[param]
    if param == 'retirement_age':
        step = max(1, int((hi - lo) / (n - 1)))
        return [float(a) for a in range(int(lo), int(hi) + 1, step)]
    if param in LOG_SCALE_PARAMS:
        return list(np.geomspace(lo, hi, n))
    return list(np.linspace(lo, hi, n))


def run_2d_sweep(
    config: SimulationConfig,
    param1: str,
    param1_range: List[float],
    param2: str,
    param2_range: List[float],
    spending_rule: SpendingRule | None = None,
    n_simulations: int = 200,
    metric: str = 'mean_utility',
    portfolio_idx: int = 2,
) -> Dict:
    """Sweep two parameters, return 2D grid of metric values."""
    if spending_rule is None:
        if config.vitality_floor < 1.0:
            spending_rule = MarginalUtilitySpending()
        else:
            spending_rule = AmortizedSpending()

    seeds = _generate_seeds(n_simulations)
    grid = np.zeros((len(param1_range), len(param2_range)))

    total_cells = len(param1_range) * len(param2_range)
    cell = 0

    for i, v1 in enumerate(param1_range):
        for j, v2 in enumerate(param2_range):
            cell += 1
            overrides = {
                param1: int(v1) if param1 == 'retirement_age' else v1,
                param2: int(v2) if param2 == 'retirement_age' else v2,
            }
            cfg = replace(config, **overrides)
            scorer = CRRAUtility.from_config(cfg)
            sim_years = cfg.expected_lifespan - cfg.start_age + 1

            batch = _run_sim_batch(cfg, spending_rule, seeds)

            utilities = []
            ruin_count = 0

            for result, result2 in batch:
                u = _score_utility(scorer, result, result2, portfolio_idx, cfg)
                utilities.append(u)
                history = result.portfolios.portfolios[portfolio_idx].get_nw_history()
                if min(history) < 0:
                    ruin_count += 1

            u_arr = np.array(utilities)
            mean_u = float(np.mean(u_arr))
            median_u = float(np.median(u_arr))

            if metric == 'mean_ce':
                grid[i, j] = scorer.certainty_equivalent(mean_u, sim_years)
            elif metric == 'median_ce':
                grid[i, j] = scorer.certainty_equivalent(median_u, sim_years)
            elif metric == 'ruin_pct':
                grid[i, j] = ruin_count / n_simulations
            else:
                grid[i, j] = mean_u

            if cell % 5 == 0 or cell == total_cells:
                print(f"  [{cell}/{total_cells}] {param1}={v1:.2f}, "
                      f"{param2}={v2:.2f} -> {metric}={grid[i, j]:.4f}")

    return {
        'param1': param1, 'param1_range': param1_range,
        'param2': param2, 'param2_range': param2_range,
        'grid': grid, 'metric': metric,
    }


# ---------------------------------------------------------------------------
# Instrument Comparison
# ---------------------------------------------------------------------------

INSTRUMENT_NAMES = ['generic', 'futures', 'box_spread']


def run_instrument_comparison(
    config: SimulationConfig,
    spending_rule: SpendingRule | None = None,
    utility_scorer: UtilityScorer | None = None,
    n_simulations: int = 500,
    leverage_range: List[float] | None = None,
) -> Dict[str, Dict]:
    """Compare leverage instruments across leverage levels using CRN.

    Runs a leverage sweep for each instrument type (generic, futures, box_spread)
    with the same random seeds for fair comparison.

    Returns dict keyed by instrument name, each containing leverage sweep results.
    """
    if spending_rule is None:
        if config.vitality_floor < 1.0:
            spending_rule = MarginalUtilitySpending()
        else:
            spending_rule = AmortizedSpending()
    if utility_scorer is None:
        utility_scorer = CRRAUtility.from_config(config)

    results: Dict[str, Dict] = {}

    for inst in INSTRUMENT_NAMES:
        print(f"\n--- Instrument: {inst} ---")
        cfg = replace(config, leverage_instrument=inst)
        scorer = CRRAUtility.from_config(cfg)
        sweep = run_leverage_sweep(
            cfg, spending_rule, scorer,
            leverage_range=leverage_range,
            n_simulations=n_simulations,
        )
        results[inst] = sweep

    return results
