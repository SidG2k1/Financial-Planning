# Retirement financial planning simulator — CLI entry point
# All monetary values are in present-dollar terms, 1 unit = $1,000 (2023 USD)
#
# Architecture: Financial Model (simulation) + Utility Model (scoring)
#               + Decision Model (spending rules, sweeps)

import argparse

from config import SimulationConfig, format_money, load_env
from models import compute_optimal_leverage
from plotting import (
    plot_2d_sweep,
    plot_leverage_sweep,
    plot_monte_carlo,
    plot_results,
    plot_retirement_sweep,
)
from simulator import run_simulation
from spending import (
    AmortizedSpending,
    FixedSpending,
    VitalityAmortizedSpending,
)
from sweeps import (
    PARAM_MAP,
    PORTFOLIO_NAME_MAP,
    _make_param_range,
    run_2d_sweep,
    run_leverage_sweep,
    run_monte_carlo,
    run_retirement_sweep,
)
from utility import CRRAUtility


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Retirement financial planning simulator'
    )
    # Personal
    parser.add_argument('--start-age', type=int, default=25,
                        help='Age to start simulation (default: 25)')
    parser.add_argument('--retirement-age', type=int, default=None,
                        help='Retirement age (default: from .env or 65)')
    parser.add_argument('--lifespan', type=int, default=90,
                        help='Expected lifespan (default: 90)')
    parser.add_argument('--initial-nw', type=float, default=None,
                        help='Initial net worth in $1,000 units (default: from .env or 50)')
    parser.add_argument('--initial-expenses', type=float, default=None,
                        help='Initial annual expenses in $1,000 units (default: from .env or 40)')
    parser.add_argument('--spending-floor', type=float, default=None,
                        help='Minimum annual spending in $1,000 units (default: from .env or 30)')
    # Market model
    parser.add_argument('--erp', type=float, default=0.025,
                        help='Equity risk premium (default: 0.025)')
    parser.add_argument('--stock-vol', type=float, default=0.10,
                        help='Annualized equity volatility (default: 0.10)')
    parser.add_argument('--bond-yield', type=float, default=0.02,
                        help='Initial real bond yield (default: 0.02)')
    parser.add_argument('--margin-spread', type=float, default=0.015,
                        help='Broker spread above bond yield (default: 0.015)')
    # Leverage
    parser.add_argument('--leverage', type=float, default=2.0,
                        help='Leverage ratio for leveraged portfolio (default: 2.0)')
    parser.add_argument('--optimal-leverage', action='store_true',
                        help='Use Kelly-optimal leverage instead of --leverage')
    # Simulation mode
    parser.add_argument('--monte-carlo', type=int, default=0, metavar='N',
                        help='Run N Monte Carlo simulations (default: 0 = single run)')
    parser.add_argument('--leverage-sweep', type=int, default=0, metavar='N',
                        help='Run leverage sweep with N sims per level')
    parser.add_argument('--retirement-sweep', type=int, default=0, metavar='N',
                        help='Sweep retirement ages with N sims per age')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    # Decision model
    parser.add_argument('--amortized', action='store_true',
                        help='Use lifecycle amortized spending instead of fixed')
    # Utility model
    parser.add_argument('--discount-rate', type=float, default=0.03,
                        help='Annual time discount rate for utility (default: 0.03)')
    parser.add_argument('--utility-power', type=float, default=0.65,
                        help='CRRA exponent: U(c) = c^alpha (default: 0.65)')
    parser.add_argument('--fire-multiplier', type=float, default=1.8,
                        help='Utility multiplier for retirement years (default: 1.8)')
    # Vitality curve
    parser.add_argument('--vitality-peak', type=int, default=30,
                        help='Age of peak vitality (default: 30)')
    parser.add_argument('--vitality-half-life', type=float, default=35.0,
                        help='Vitality half-life in years from peak (default: 35)')
    parser.add_argument('--vitality-floor', type=float, default=0.3,
                        help='Minimum vitality multiplier (default: 0.3)')
    parser.add_argument('--no-vitality', action='store_true',
                        help='Disable vitality weighting (floor=1.0)')
    # Social Security
    parser.add_argument('--no-ss', action='store_true',
                        help='Disable Social Security')
    parser.add_argument('--ss-claiming-age', type=int, default=67,
                        help='SS claiming age (default: 67, range 62-70)')
    # Tax
    parser.add_argument('--retirement-tax-advantage', type=float, default=1.25,
                        help='Retirement withdrawal purchasing power multiplier '
                             '(default: 1.25, i.e. LTCG/Roth vs earned income)')
    # Margin calls
    parser.add_argument('--maintenance-margin', type=float, default=0.25,
                        help='Margin call equity threshold (default: 0.25)')
    parser.add_argument('--no-margin-calls', action='store_true',
                        help='Disable margin call mechanics')
    # MC variance reduction
    parser.add_argument('--antithetic', action='store_true',
                        help='Enable antithetic variates for MC variance reduction')
    # Stochastic lifespan
    parser.add_argument('--stochastic-lifespan', action='store_true',
                        help='Enable Gompertz stochastic lifespan')
    parser.add_argument('--gompertz-a', type=float, default=0.00003,
                        help='Gompertz baseline hazard (default: 0.00003)')
    parser.add_argument('--gompertz-b', type=float, default=0.085,
                        help='Gompertz aging rate (default: 0.085)')
    parser.add_argument('--max-age', type=int, default=110,
                        help='Hard upper bound for simulation (default: 110)')
    # Stochastic income
    parser.add_argument('--stochastic-income', action='store_true',
                        help='Enable market-correlated job loss')
    parser.add_argument('--job-loss-prob', type=float, default=0.03,
                        help='Annual baseline job loss probability (default: 0.03)')
    parser.add_argument('--job-loss-sensitivity', type=float, default=5.0,
                        help='Market crash amplification of job loss (default: 5.0)')
    # Bayesian parameter uncertainty
    parser.add_argument('--bayesian', action='store_true',
                        help='Enable Bayesian parameter uncertainty')
    parser.add_argument('--bayesian-erp-std', type=float, default=0.02,
                        help='Posterior std of ERP (default: 0.02)')
    parser.add_argument('--bayesian-vol-std', type=float, default=0.30,
                        help='Posterior std of log(vol) (default: 0.30)')
    parser.add_argument('--bayesian-bond-std', type=float, default=0.01,
                        help='Posterior std of bond yield (default: 0.01)')
    # 2D sweep
    parser.add_argument('--sweep-2d', nargs=3, metavar=('PARAM1', 'PARAM2', 'N'),
                        help='2D parameter sweep (e.g., --sweep-2d retirement-age leverage 200)')
    parser.add_argument('--sweep-metric', default='mean_utility',
                        choices=['mean_utility', 'mean_ce', 'median_ce', 'ruin_pct'],
                        help='Metric to optimize in 2D sweep (default: mean_utility)')
    parser.add_argument('--sweep-portfolio', default='leveraged',
                        choices=['cash', 'stock', 'leveraged'],
                        help='Portfolio to score in 2D sweep (default: leveraged)')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = load_env()

    def _env_or(cli_val, env_key, default):
        """CLI arg (if set) > .env > hardcoded default."""
        if cli_val is not None:
            return cli_val
        return env.get(env_key, default)

    config = SimulationConfig(
        start_age=args.start_age,
        retirement_age=_env_or(args.retirement_age, 'retirement_age', 65),
        expected_lifespan=args.lifespan,
        initial_net_worth=_env_or(args.initial_nw, 'initial_net_worth', 50),
        initial_expenses=_env_or(args.initial_expenses, 'initial_expenses', 40),
        spending_floor=_env_or(args.spending_floor, 'spending_floor', 30),
        income_schedule=env.get('income_schedule', [75] * 5 + [100] * 10 + [120] * 25),
        one_time_expenses=env.get('one_time_expenses', {}),
        state_tax_rate=env.get('state_tax_rate', 0.05),
        equity_risk_premium=args.erp,
        stock_vol=args.stock_vol,
        initial_bond_yield=args.bond_yield,
        long_run_bond_yield=args.bond_yield,
        margin_spread=args.margin_spread,
        leverage_ratio=args.leverage,
        utility_power=args.utility_power,
        discount_rate=args.discount_rate,
        fire_multiplier=args.fire_multiplier,
        vitality_peak_age=args.vitality_peak,
        vitality_half_life=args.vitality_half_life,
        vitality_floor=1.0 if args.no_vitality else args.vitality_floor,
        ss_enabled=not args.no_ss,
        ss_claiming_age=args.ss_claiming_age,
        retirement_tax_advantage=args.retirement_tax_advantage,
        maintenance_margin=0.0 if args.no_margin_calls else args.maintenance_margin,
        antithetic=args.antithetic,
        stochastic_lifespan=args.stochastic_lifespan,
        gompertz_a=args.gompertz_a,
        gompertz_b=args.gompertz_b,
        max_age=args.max_age,
        stochastic_income=args.stochastic_income,
        job_loss_base_prob=args.job_loss_prob,
        job_loss_market_sensitivity=args.job_loss_sensitivity,
        bayesian=args.bayesian,
        bayesian_erp_std=args.bayesian_erp_std,
        bayesian_vol_std=args.bayesian_vol_std,
        bayesian_bond_yield_std=args.bayesian_bond_std,
    )

    if args.optimal_leverage:
        config.leverage_ratio = compute_optimal_leverage(config)
        print(f"Using Kelly-optimal leverage: {config.leverage_ratio:.2f}x")

    # Build spending rule (Decision Model)
    if args.amortized or args.retirement_sweep > 0 or args.sweep_2d:
        if config.vitality_floor < 1.0:
            spending_rule = VitalityAmortizedSpending()
        else:
            spending_rule = AmortizedSpending()
    else:
        spending_rule = FixedSpending(config.initial_expenses, config.lifestyle_inflation)

    # Build utility scorer (Utility Model)
    utility_scorer = CRRAUtility.from_config(config)

    sim_years = config.expected_lifespan - config.start_age + 1

    if args.sweep_2d:
        p1_cli, p2_cli, n_str = args.sweep_2d
        n_sims = int(n_str)
        p1 = PARAM_MAP[p1_cli]
        p2 = PARAM_MAP[p2_cli]
        p1_range = _make_param_range(p1)
        p2_range = _make_param_range(p2)
        portfolio_idx = PORTFOLIO_NAME_MAP[args.sweep_portfolio]
        print(f"Running 2D sweep: {p1} x {p2} "
              f"({len(p1_range)}x{len(p2_range)} grid, "
              f"{n_sims} sims/point)...")
        sweep = run_2d_sweep(
            config, p1, p1_range, p2, p2_range,
            spending_rule=spending_rule, n_simulations=n_sims,
            metric=args.sweep_metric, portfolio_idx=portfolio_idx)
        plot_2d_sweep(sweep, config)
    elif args.retirement_sweep > 0:
        print(f"Running retirement age sweep "
              f"({args.retirement_sweep} sims per age)...")
        vit_desc = (f"vitality(peak={config.vitality_peak_age}, "
                    f"hl={config.vitality_half_life:.0f}, "
                    f"floor={config.vitality_floor})")
        ss_desc = (f"SS@{config.ss_claiming_age}" if config.ss_enabled
                   else "no SS")
        print(f"FIRE={config.fire_multiplier}, "
              f"U(c)=c^{config.utility_power}, "
              f"delta={config.discount_rate:.0%}, {vit_desc}")
        print(f"{ss_desc}, "
              f"retirement_tax_adv={config.retirement_tax_advantage:.0%}")
        print()
        sweep_results = run_retirement_sweep(
            config, spending_rule,
            n_simulations=args.retirement_sweep)
        plot_retirement_sweep(sweep_results, config)
    elif args.leverage_sweep > 0:
        print(f"Running leverage sweep ({args.leverage_sweep} sims per level)...")
        kelly = compute_optimal_leverage(config)
        print(f"Kelly optimal: {kelly:.2f}x  |  Half-Kelly: {kelly/2:.2f}x")
        print(f"Spending: {type(spending_rule).__name__}")
        print()
        sweep = run_leverage_sweep(config, spending_rule, utility_scorer,
                                   n_simulations=args.leverage_sweep)
        plot_leverage_sweep(sweep, config)
    elif args.monte_carlo > 0:
        print(f"Running {args.monte_carlo} Monte Carlo simulations...")
        print(f"Spending: {type(spending_rule).__name__}")
        results = run_monte_carlo(config, spending_rule, utility_scorer,
                                  n_simulations=args.monte_carlo)
        plot_monte_carlo(results, config)
    else:
        result = run_simulation(config, spending_rule, seed=args.seed)
        vfloor = config.vitality_floor
        print(f"\n=== Lifetime Utility (U(c) = c^{config.utility_power}, "
              f"delta={config.discount_rate:.0%}, FIRE={config.fire_multiplier}, "
              f"vitality floor={vfloor}) ===")
        print(f"{'Portfolio':<30s}  {'E[U]':>10s}  {'CE':>14s}")
        print("-" * 58)
        for j, p in enumerate(result.portfolios.portfolios):
            u = utility_scorer.score(result.spending[j], config)
            ce = utility_scorer.certainty_equivalent(u, sim_years)
            print(f"{type(p).__name__:<30s}  {u:>10,.1f}  ({format_money(ce)}/yr)")
        print()
        plot_results(result, config)


if __name__ == '__main__':
    main()
