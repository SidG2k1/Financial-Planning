# Visualization: all plot_* functions with lazy matplotlib import

from __future__ import annotations

from typing import Dict

import numpy as np

from config import SimulationConfig, SimulationResult, format_money
from sweeps import LOG_SCALE_PARAMS


def _get_plt():
    """Lazy import of matplotlib to avoid crashes in headless environments."""
    import matplotlib.pyplot as plt
    return plt


def plot_results(result: SimulationResult, config: SimulationConfig) -> None:
    """Plot net worth history for all portfolios."""
    plt = _get_plt()
    portfolios = result.portfolios
    histories = portfolios.get_nw_history()
    ages = list(range(config.start_age, config.start_age + len(histories[0])))

    for i, portfolio in enumerate(portfolios.portfolios):
        values = [h / 1000 for h in histories[i]]
        plt.plot(ages, values, label=type(portfolio).__name__)

    plt.legend()
    plt.xlabel('Age')
    plt.ylabel('Net Worth ($M)')
    plt.axvline(x=config.retirement_age, color='r', linestyle='--')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Retirement Portfolio Simulation')
    plt.tight_layout()
    plt.show()


def plot_monte_carlo(results: Dict[str, Dict], config: SimulationConfig) -> None:
    """Plot Monte Carlo results with shaded confidence bands."""
    plt = _get_plt()
    n_portfolios = len(results)
    fig, axes = plt.subplots(1, n_portfolios,
                             figsize=(6 * n_portfolios, 5), sharey=True)
    if n_portfolios == 1:
        axes = [axes]

    for ax, (name, data) in zip(axes, results.items()):
        ages = data['ages']
        ax.fill_between(ages, data['p10'], data['p90'],
                        alpha=0.15, label='10th-90th %ile')
        ax.fill_between(ages, data['p25'], data['p75'],
                        alpha=0.3, label='25th-75th %ile')
        ax.plot(ages, data['median'], linewidth=2, label='Median')
        ax.axvline(x=config.retirement_age, color='r', linestyle='--', alpha=0.5)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        u = data.get('mean_utility', 0)
        ce = data.get('mean_ce', 0)
        ax.set_title(f'{name}\nE[U]={u:,.1f}  (CE={format_money(ce)}/yr)')
        ax.set_xlabel('Age')
        ax.legend(fontsize=8)

    axes[0].set_ylabel('Net Worth ($M)')
    fig.suptitle('Monte Carlo Retirement Simulation')
    plt.tight_layout()
    plt.show()

    # Print utility summary
    print(f"\n=== Utility Summary ===")
    print(f"{'Portfolio':<28s}  {'E[U]':>10s}  {'CE':>14s}")
    print("-" * 56)
    for name, data in results.items():
        print(f"{name:<28s}  {data['mean_utility']:>10,.1f}  "
              f"({format_money(data['mean_ce'])}/yr)")
    print()


def plot_leverage_sweep(sweep: Dict[str, list], config: SimulationConfig) -> None:
    """Plot leverage sweep: NW percentiles, ruin probability, and CE."""
    plt = _get_plt()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    levs = sweep['leverage']

    ax1.fill_between(levs, sweep['p10_nw'], sweep['p90_nw'],
                     alpha=0.15, label='10th-90th %ile')
    ax1.fill_between(levs, sweep['p25_nw'], sweep['p75_nw'],
                     alpha=0.3, label='25th-75th %ile')
    ax1.plot(levs, sweep['median_nw'], 'o-', linewidth=2, label='Median')
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Leverage Ratio')
    ax1.set_ylabel('Final Net Worth ($M)')
    ax1.set_title('Final NW vs Leverage')
    ax1.set_yscale('symlog', linthresh=100)
    ax1.legend(fontsize=8)

    ax2.plot(levs, [r * 100 for r in sweep['ruin_pct']], 'o-',
             linewidth=2, color='red')
    ax2.axhline(y=5, color='orange', linestyle='--', alpha=0.7,
                label='5% risk threshold')
    ax2.set_xlabel('Leverage Ratio')
    ax2.set_ylabel('Ruin Probability (%)')
    ax2.set_title('Ruin Risk vs Leverage')
    ax2.set_ylim(bottom=0)
    ax2.legend(fontsize=8)

    u_vals = sweep['mean_utility']
    ax3.plot(levs, u_vals, 'o-', linewidth=2, color='purple')
    best_idx = int(np.argmax(u_vals))
    ax3.axvline(x=levs[best_idx], color='purple', linestyle=':',
                alpha=0.7, label=f'Max E[U] ({levs[best_idx]:.2f}x)')
    ax3.set_xlabel('Leverage Ratio')
    ax3.set_ylabel('E[U] (Total Lifetime Utility)')
    ax3.set_title(f'Expected Utility (U=c^{config.utility_power})')
    ax3.legend(fontsize=8)

    fig.suptitle(f'Leverage Sweep (age {config.start_age}-{config.expected_lifespan}, '
                 f'retire {config.retirement_age})')
    plt.tight_layout()
    plt.show()


def plot_retirement_sweep(
    results: Dict[str, Dict],
    config: SimulationConfig,
) -> None:
    """Plot E[U] vs retirement age for each portfolio, with CE reference."""
    plt = _get_plt()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for name, data in results.items():
        ages = data['retirement_ages']
        utils = data['mean_utilities']
        ax1.plot(ages, utils, 'o-', linewidth=2, label=name, markersize=4)

        best_idx = int(np.argmax(utils))
        opt_age = ages[best_idx]
        opt_u = utils[best_idx]
        ax1.annotate(f'{opt_age}',
                     xy=(opt_age, opt_u),
                     xytext=(5, 10), textcoords='offset points',
                     fontsize=9, fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color='red'))

    ax1.set_xlabel('Retirement Age')
    ax1.set_ylabel('E[U] (Total Lifetime Utility)')
    vf = config.vitality_floor
    ss_lbl = f"SS@{config.ss_claiming_age}" if config.ss_enabled else "no SS"
    ax1.set_title(f'E[U] vs Retirement Age\n'
                  f'(FIRE={config.fire_multiplier}, vit={vf}, {ss_lbl}, '
                  f'tax_adv={config.retirement_tax_advantage:.0%})')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Right panel: CE for reference
    for name, data in results.items():
        ages = data['retirement_ages']
        ces = [c * 1000 for c in data['mean_ces']]
        ax2.plot(ages, ces, 'o-', linewidth=2, label=name, markersize=4)
    ax2.set_xlabel('Retirement Age')
    ax2.set_ylabel('CE ($k/yr)')
    ax2.set_title('Certainty Equivalent (for reference)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Retirement Age Optimization (Amortized Spending)')
    plt.tight_layout()
    plt.show()

    # Summary table
    print(f"\n=== Retirement Age Sweep Summary ===")
    ss_desc = f"SS@{config.ss_claiming_age}" if config.ss_enabled else "no SS"
    print(f"FIRE={config.fire_multiplier}, "
          f"U(c)=c^{config.utility_power}, "
          f"delta={config.discount_rate:.0%}, "
          f"vitality(peak={config.vitality_peak_age}, "
          f"hl={config.vitality_half_life:.0f}, "
          f"floor={config.vitality_floor})")
    print(f"{ss_desc}, "
          f"retirement_tax_adv={config.retirement_tax_advantage:.0%}, "
          f"lifespan={config.expected_lifespan}")
    print(f"{'Portfolio':<28s}  {'Optimal Age':>11s}  {'E[U]':>10s}  {'CE':>14s}")
    print("-" * 68)
    for name, data in results.items():
        best_idx = int(np.argmax(data['mean_utilities']))
        print(f"{name:<28s}  {data['optimal_age']:>11d}  "
              f"{data['mean_utilities'][best_idx]:>10,.1f}  "
              f"({format_money(data['optimal_ce'])}/yr)")
    print()


def plot_2d_sweep(sweep: Dict, config: SimulationConfig) -> None:
    """Plot 2D parameter sweep as filled contour chart."""
    plt = _get_plt()
    X, Y = np.meshgrid(sweep['param2_range'], sweep['param1_range'])
    is_ruin = sweep['metric'] == 'ruin_pct'
    is_ce = sweep['metric'] in ('mean_ce', 'median_ce')
    grid = sweep['grid'] * 1000 if is_ce else sweep['grid']

    fig, ax = plt.subplots(figsize=(10, 8))

    if sweep['param1'] in LOG_SCALE_PARAMS:
        ax.set_yscale('symlog', linthresh=100)
    if sweep['param2'] in LOG_SCALE_PARAMS:
        ax.set_xscale('symlog', linthresh=100)

    levels = 20
    cmap = 'RdYlGn_r' if is_ruin else 'RdYlGn'
    cf = ax.contourf(X, Y, grid, levels=levels, cmap=cmap)
    cs = ax.contour(X, Y, grid, levels=10, colors='black', linewidths=0.5, alpha=0.5)
    fmt = '%.1f%%' if is_ruin else ('$%.0fk' if is_ce else '%.0f')
    ax.clabel(cs, inline=True, fontsize=8, fmt=fmt)

    # Mark optimum
    opt_idx = np.unravel_index(
        np.argmin(grid) if is_ruin else np.argmax(grid), grid.shape)
    ax.plot(sweep['param2_range'][opt_idx[1]], sweep['param1_range'][opt_idx[0]],
            'w*', markersize=15, markeredgecolor='black')

    if is_ruin:
        label = f'{sweep["metric"]} (%)'
    elif is_ce:
        label = f'{sweep["metric"]} ($k/yr)'
    else:
        label = f'E[U] (Total Lifetime Utility)'
    fig.colorbar(cf, label=label)
    ax.set_xlabel(sweep['param2'])
    ax.set_ylabel(sweep['param1'])
    ax.set_title(f'2D Sweep: {sweep["param1"]} vs {sweep["param2"]}\n'
                 f'Optimum at {sweep["param1"]}='
                 f'{sweep["param1_range"][opt_idx[0]]:.2f}, '
                 f'{sweep["param2"]}='
                 f'{sweep["param2_range"][opt_idx[1]]:.2f}')
    plt.tight_layout()
    plt.show()


def plot_instrument_comparison(
    results: Dict[str, Dict],
    config: SimulationConfig,
) -> None:
    """Plot instrument comparison: CE, ruin %, and E[U] vs leverage for each instrument."""
    plt = _get_plt()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    colors = {'generic': 'gray', 'futures': '#2196F3', 'box_spread': '#4CAF50'}
    labels = {'generic': 'Generic', 'futures': 'Futures (1256)',
              'box_spread': 'Box Spread + ETF'}

    for inst_name, sweep in results.items():
        levs = sweep['leverage']
        c = colors.get(inst_name, 'black')
        lbl = labels.get(inst_name, inst_name)

        # Panel 1: CE vs leverage
        ces = [ce * 1000 for ce in sweep['mean_ce']]
        ax1.plot(levs, ces, 'o-', color=c, linewidth=2,
                 label=lbl, markersize=4)

        # Panel 2: Ruin probability
        ax2.plot(levs, [r * 100 for r in sweep['ruin_pct']], 'o-', color=c,
                 linewidth=2, label=lbl, markersize=4)

        # Panel 3: E[U]
        ax3.plot(levs, sweep['mean_utility'], 'o-', color=c,
                 linewidth=2, label=lbl, markersize=4)

    ax1.set_xlabel('Leverage Ratio')
    ax1.set_ylabel('Certainty Equivalent ($k/yr)')
    ax1.set_title('CE vs Leverage by Instrument')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Leverage Ratio')
    ax2.set_ylabel('Ruin Probability (%)')
    ax2.set_title('Ruin Risk vs Leverage')
    ax2.set_ylim(bottom=0)
    ax2.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='5% threshold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax3.set_xlabel('Leverage Ratio')
    ax3.set_ylabel('E[U] (Total Lifetime Utility)')
    ax3.set_title(f'Expected Utility (U=c^{config.utility_power})')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    fig.suptitle(f'Leverage Instrument Comparison (retire@{config.retirement_age}, '
                 f'age {config.start_age}-{config.expected_lifespan})')
    plt.tight_layout()
    plt.show()

    # Summary table
    print(f"\n=== Instrument Comparison Summary ===")
    print(f"{'Instrument':<20s}  {'Best Lev':>8s}  {'E[U]':>10s}  {'CE':>14s}  {'Ruin':>6s}")
    print("-" * 64)
    for inst_name, sweep in results.items():
        best_idx = int(np.argmax(sweep['mean_utility']))
        lbl = labels.get(inst_name, inst_name)
        print(f"{lbl:<20s}  {sweep['leverage'][best_idx]:>7.2f}x  "
              f"{sweep['mean_utility'][best_idx]:>10,.1f}  "
              f"({format_money(sweep['mean_ce'][best_idx])}/yr)  "
              f"{sweep['ruin_pct'][best_idx]:>5.1%}")
    print()
