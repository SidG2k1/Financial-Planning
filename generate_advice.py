"""Generate financial advisor charts and analysis. Saves all figures to assets/."""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import replace

from config import SimulationConfig, format_money, load_env
from models import compute_optimal_leverage, vitality_at_age
from simulator import run_simulation
from spending import (
    AmortizedSpending,
    FixedSpending,
    VitalityAmortizedSpending,
)
from sweeps import (
    _generate_seeds,
    _run_sim_batch,
    _score_utility,
    run_leverage_sweep,
    evaluate_point,
    run_monte_carlo,
    run_retirement_sweep,
    run_2d_sweep,
)
from utility import CRRAUtility

ASSETS = os.path.join(os.path.dirname(__file__), 'assets')

config = SimulationConfig(
    **load_env(),
    bayesian=True,
    stochastic_income=True,
    antithetic=True,
)
kelly = compute_optimal_leverage(config)

print("=" * 70)
print("  FINANCIAL PLANNING ANALYSIS")
inc = config.income_schedule
inc_lo, inc_hi = min(inc), max(inc)
inc_str = f"${inc_lo:.0f}-{inc_hi:.0f}k" if inc_lo != inc_hi else f"${inc_lo:.0f}k"
print(f"  Profile: age {config.start_age}, income {inc_str}, NW ${config.initial_net_worth:.0f}k")
print(f"  Expenses: ${config.initial_expenses}k/yr  |  Risk tolerance: AGGRESSIVE")
print(f"  Kelly optimal leverage: {kelly:.2f}x")
print("=" * 70)


# -----------------------------------------------------------------------
# Helper: format utility for display
# -----------------------------------------------------------------------
def fmt_u(u: float) -> str:
    """Format a total utility value for display."""
    return f"{u:,.1f}"


# -----------------------------------------------------------------------
# 1. Retirement Age Sweep — find the optimal retirement age
# -----------------------------------------------------------------------
print("\n[1/7] Running retirement age sweep (300 sims/age)...")

spending_rule = VitalityAmortizedSpending()
ret_results = run_retirement_sweep(config, spending_rule, n_simulations=300)

# Find optimal age per portfolio (by max mean utility — monotonic with CE)
for name, data in ret_results.items():
    best_idx = int(np.argmax(data['mean_utilities']))
    data['optimal_age'] = data['retirement_ages'][best_idx]
    data['optimal_utility'] = data['mean_utilities'][best_idx]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for name, data in ret_results.items():
    ages = data['retirement_ages']
    utils = data['mean_utilities']
    ax1.plot(ages, utils, 'o-', linewidth=2, label=name, markersize=4)
    opt_age = data['optimal_age']
    opt_u = data['optimal_utility']
    ax1.annotate(f'{opt_age}', xy=(opt_age, opt_u),
                 xytext=(5, 10), textcoords='offset points',
                 fontsize=9, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='red'))

ax1.set_xlabel('Retirement Age')
ax1.set_ylabel('E[U] (Total Lifetime Utility)')
ss_lbl = f"SS@{config.ss_claiming_age}" if config.ss_enabled else "no SS"
ax1.set_title(f'E[U] vs Retirement Age\n'
              f'(FIRE={config.fire_multiplier}, vit_floor={config.vitality_floor}, '
              f'{ss_lbl}, tax_adv={config.retirement_tax_advantage:.0%})')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Right panel: CE (for reference)
for name, data in ret_results.items():
    ages = data['retirement_ages']
    ces = [c * 1000 for c in data['mean_ces']]
    ax2.plot(ages, ces, 'o-', linewidth=2, label=name, markersize=4)
ax2.set_xlabel('Retirement Age')
ax2.set_ylabel('CE ($k/yr)')
ax2.set_title('Certainty Equivalent (for reference)')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

fig.suptitle('Retirement Age Optimization (Vitality-Weighted Amortized Spending)')
plt.tight_layout()
fig.savefig(os.path.join(ASSETS, '1_retirement_sweep.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

print("\n  === Retirement Sweep Summary ===")
for name, data in ret_results.items():
    print(f"  {name:<28s}  optimal retire @ {data['optimal_age']}  "
          f"E[U]={fmt_u(data['optimal_utility'])}  "
          f"(CE={format_money(data['optimal_ce'])}/yr)")


# -----------------------------------------------------------------------
# 2. Leverage Sweep — find optimal leverage
# -----------------------------------------------------------------------
print("\n[2/7] Running leverage sweep (300 sims/level)...")

lev_spending = VitalityAmortizedSpending()
scorer = CRRAUtility.from_config(config)
lev_results = run_leverage_sweep(config, lev_spending, scorer,
                                  leverage_range=[1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0],
                                  n_simulations=300)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
levs = lev_results['leverage']

ax1.fill_between(levs, lev_results['p10_nw'], lev_results['p90_nw'],
                 alpha=0.15, label='10th-90th %ile')
ax1.fill_between(levs, lev_results['p25_nw'], lev_results['p75_nw'],
                 alpha=0.3, label='25th-75th %ile')
ax1.plot(levs, lev_results['median_nw'], 'o-', linewidth=2, label='Median')
ax1.axvline(x=kelly, color='green', linestyle='--', alpha=0.7,
            label=f'Kelly optimal ({kelly:.2f}x)')
ax1.axhline(y=0, color='r', linestyle='--', alpha=0.3)
ax1.set_xlabel('Leverage Ratio')
ax1.set_ylabel('Final Net Worth ($M)')
ax1.set_title('Final NW vs Leverage')
ax1.set_yscale('symlog', linthresh=1)
ax1.legend(fontsize=8)

ax2.plot(levs, [r * 100 for r in lev_results['ruin_pct']], 'o-',
         linewidth=2, color='red')
ax2.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='5% threshold')
ax2.axvline(x=kelly, color='green', linestyle='--', alpha=0.7,
            label=f'Kelly ({kelly:.2f}x)')
ax2.set_xlabel('Leverage Ratio')
ax2.set_ylabel('Ruin Probability (%)')
ax2.set_title('Ruin Risk vs Leverage')
ax2.set_ylim(bottom=0)
ax2.legend(fontsize=8)

u_vals = lev_results['mean_utility']
best_idx = int(np.argmax(u_vals))
ax3.plot(levs, u_vals, 'o-', linewidth=2, color='purple')
ax3.axvline(x=kelly, color='green', linestyle='--', alpha=0.7,
            label=f'Kelly ({kelly:.2f}x)')
ax3.axvline(x=levs[best_idx], color='purple', linestyle=':',
            alpha=0.7, label=f'Max E[U] ({levs[best_idx]:.2f}x)')
ax3.set_xlabel('Leverage Ratio')
ax3.set_ylabel('E[U] (Total Lifetime Utility)')
ax3.set_title(f'Expected Utility (U=c^{config.utility_power})')
ax3.legend(fontsize=8)

fig.suptitle(f'Leverage Sweep (age {config.start_age}-{config.expected_lifespan}, retire {config.retirement_age})')
plt.tight_layout()
fig.savefig(os.path.join(ASSETS, '2_leverage_sweep.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

best_lev = levs[best_idx]
best_u = u_vals[best_idx]
best_ce = lev_results['mean_ce'][best_idx]
best_ruin = lev_results['ruin_pct'][best_idx]
print(f"\n  Utility-maximizing leverage: {best_lev:.2f}x  "
      f"E[U]={fmt_u(best_u)}  (CE={format_money(best_ce)}/yr)  ruin={best_ruin:.1%}")
print(f"  Kelly optimal: {kelly:.2f}x  |  Half-Kelly: {kelly/2:.2f}x")


# -----------------------------------------------------------------------
# 3. 2D Sweep: retirement age vs leverage — joint optimization
# -----------------------------------------------------------------------

def adam_optimize(cfg, spending_rule, n_sims=100, steps=25, lr=2.0,
                  beta1=0.9, beta2=0.999, eps=1e-8):
    """Adam optimizer over (retirement_age, leverage) via finite-difference gradients."""
    seeds = _generate_seeds(n_sims)
    ret, lev = 50.0, 2.0  # start near center
    m = np.zeros(2)
    v = np.zeros(2)
    path = []
    h_ret, h_lev = 2.0, 0.2

    for t in range(1, steps + 1):
        u_ret_p = evaluate_point(cfg, ret + h_ret, lev, spending_rule, seeds)
        u_ret_m = evaluate_point(cfg, ret - h_ret, lev, spending_rule, seeds)
        u_lev_p = evaluate_point(cfg, ret, lev + h_lev, spending_rule, seeds)
        u_lev_m = evaluate_point(cfg, ret, lev - h_lev, spending_rule, seeds)

        grad = np.array([
            (u_ret_p - u_ret_m) / (2 * h_ret),
            (u_lev_p - u_lev_m) / (2 * h_lev),
        ])

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

        ret += lr * m_hat[0] / (np.sqrt(v_hat[0]) + eps)
        lev += lr * m_hat[1] / (np.sqrt(v_hat[1]) + eps)
        ret = float(np.clip(ret, 30, 70))
        lev = float(np.clip(lev, 1.0, 4.0))

        u_cur = evaluate_point(cfg, ret, lev, spending_rule, seeds)
        path.append((ret, lev, u_cur))
        print(f"    Adam step {t:2d}: retire@{ret:.1f}, lev={lev:.2f}x, E[U]={u_cur:.1f}")

    return path


print("\n[3/7] Running 2D sweep (20 sims/point) + Adam optimizer (100 sims/eval)...")

sweep_2d = run_2d_sweep(
    config,
    'retirement_age', [float(a) for a in range(30, 71, 10)],
    'leverage_ratio', list(np.linspace(1.0, 4.0, 7)),
    spending_rule=VitalityAmortizedSpending(),
    n_simulations=20,
    metric='mean_utility',
    portfolio_idx=2,
)

print("\n  Running Adam optimizer...")
adam_path = adam_optimize(config, VitalityAmortizedSpending(), n_sims=100, steps=25)

X, Y = np.meshgrid(sweep_2d['param2_range'], sweep_2d['param1_range'])
grid = sweep_2d['grid']

fig, ax = plt.subplots(figsize=(10, 8))
levels = 20
cf = ax.contourf(X, Y, grid, levels=levels, cmap='RdYlGn')
cs = ax.contour(X, Y, grid, levels=10, colors='black', linewidths=0.5, alpha=0.5)
ax.clabel(cs, inline=True, fontsize=8, fmt='%.0f')

# Grid optimum (white star)
opt_idx = np.unravel_index(np.argmax(grid), grid.shape)
opt_ret = sweep_2d['param1_range'][opt_idx[0]]
opt_lev = sweep_2d['param2_range'][opt_idx[1]]
grid_u = grid[opt_idx]
ax.plot(opt_lev, opt_ret, 'w*', markersize=15, markeredgecolor='black',
        label=f'Grid optimum ({int(opt_ret)}, {opt_lev:.1f}x)')
ax.annotate(f'E[U]={grid_u:.0f}', xy=(opt_lev, opt_ret),
            fontsize=8, color='white', fontweight='bold',
            xytext=(-10, -15), textcoords='offset points', ha='right')

# Adam path
path_lev = [p[1] for p in adam_path]
path_ret = [p[0] for p in adam_path]
ax.plot(path_lev, path_ret, '-o', color='white', markersize=4, linewidth=2,
        markeredgecolor='black', markeredgewidth=0.5, label='Adam path')
ax.annotate('start', xy=(path_lev[0], path_ret[0]), fontsize=8, color='white',
            fontweight='bold', xytext=(8, 5), textcoords='offset points')
for i in range(4, len(adam_path), 5):
    ax.annotate(str(i + 1), xy=(path_lev[i], path_ret[i]),
                fontsize=7, color='white', fontweight='bold',
                ha='center', va='bottom', xytext=(0, 5), textcoords='offset points')
# Final point
final_ret, final_lev, final_u = adam_path[-1]
ax.plot(final_lev, final_ret, '*', color='cyan', markersize=15, markeredgecolor='black',
        label=f'Adam optimum ({int(round(final_ret))}, {final_lev:.1f}x)')
ax.annotate(f'E[U]={final_u:.0f}', xy=(final_lev, final_ret),
            fontsize=8, color='cyan', fontweight='bold',
            xytext=(8, -12), textcoords='offset points')

fig.colorbar(cf, label='E[U] (Total Lifetime Utility)')
ax.set_xlabel('Leverage Ratio')
ax.set_ylabel('Retirement Age')
ax.set_title(f'Joint Optimization: Retirement Age vs Leverage\n'
             f'Adam optimum: retire@{int(round(final_ret))}, {final_lev:.1f}x leverage, '
             f'E[U]={fmt_u(final_u)}')
ax.legend(loc='lower right', fontsize=8)
plt.tight_layout()
fig.savefig(os.path.join(ASSETS, '3_2d_retirement_leverage.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# --- 3D surface plot ---
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

fig3d = plt.figure(figsize=(12, 8))
ax3d = fig3d.add_subplot(111, projection='3d')
surf = ax3d.plot_surface(X, Y, grid, cmap='RdYlGn', alpha=0.85,
                         edgecolor='black', linewidth=0.2)
# Adam path on surface
path_u = [p[2] for p in adam_path]
ax3d.plot(path_lev, path_ret, path_u, '-o', color='blue', markersize=3,
          linewidth=2, zorder=10, label='Adam path')
ax3d.scatter([final_lev], [final_ret], [final_u], color='cyan', s=120,
             marker='*', edgecolors='black', zorder=11,
             label=f'Adam opt (E[U]={final_u:.0f})')
ax3d.scatter([opt_lev], [opt_ret], [grid_u], color='white', s=120,
             marker='*', edgecolors='black', zorder=11,
             label=f'Grid opt (E[U]={grid_u:.0f})')
fig3d.colorbar(surf, label='E[U]', shrink=0.6, pad=0.1)
ax3d.set_xlabel('Leverage Ratio')
ax3d.set_ylabel('Retirement Age')
ax3d.set_zlabel('E[U]')
ax3d.set_title(f'3D Utility Surface: Retirement Age vs Leverage\n'
               f'Adam optimum: retire@{int(round(final_ret))}, {final_lev:.1f}x, '
               f'E[U]={fmt_u(final_u)}')
ax3d.legend(loc='upper left', fontsize=8)
ax3d.view_init(elev=25, azim=-50)
plt.tight_layout()
fig3d.savefig(os.path.join(ASSETS, '3a_3d_retirement_leverage.png'), dpi=150, bbox_inches='tight')
plt.close(fig3d)

print(f"\n  Adam optimum: retire@{int(round(final_ret))}, leverage={final_lev:.1f}x, "
      f"E[U]={fmt_u(final_u)}")


# -----------------------------------------------------------------------
# 4. Monte Carlo at optimal retirement age — all portfolios
# -----------------------------------------------------------------------
opt_retire = ret_results.get('LeveragedStockPortfolio', {}).get('optimal_age', 42)
mc_spending = VitalityAmortizedSpending()

print(f"\n[4/7] Monte Carlo at retire@{opt_retire} (300 sims)...")

mc_config = replace(config, retirement_age=opt_retire)
mc_scorer = CRRAUtility.from_config(mc_config)
mc_results = run_monte_carlo(mc_config, mc_spending, mc_scorer, n_simulations=300)

# Filter out CashPortfolio (goes broke at early retirement — not a serious strategy)
show_portfolios = {k: v for k, v in mc_results.items() if k != 'CashPortfolio'}
n_show = len(show_portfolios)
fig, axes = plt.subplots(2, n_show, figsize=(6 * n_show, 9))
if n_show == 1:
    axes = axes.reshape(2, 1)

for col, (name, data) in enumerate(show_portfolios.items()):
    ages = data['ages']
    u = data.get('mean_utility', 0)
    ce = data.get('mean_ce', 0)

    # Top row: Net worth
    ax_nw = axes[0, col]
    ax_nw.fill_between(ages, data['p10'], data['p90'], alpha=0.15, label='10th-90th %ile')
    ax_nw.fill_between(ages, data['p25'], data['p75'], alpha=0.3, label='25th-75th %ile')
    ax_nw.plot(ages, data['median'], linewidth=2, label='Median')
    ax_nw.axvline(x=opt_retire, color='r', linestyle='--', alpha=0.5)
    ax_nw.set_yscale('symlog', linthresh=0.1)
    ax_nw.set_title(f'{name}\nE[U]={fmt_u(u)}  (CE={format_money(ce)}/yr)')
    ax_nw.legend(fontsize=7)
    if col == 0:
        ax_nw.set_ylabel('Net Worth ($M, log scale)')

    # Bottom row: Spending
    ax_sp = axes[1, col]
    ax_sp.fill_between(ages, data['spend_p10'], data['spend_p90'],
                        alpha=0.15, label='10th-90th %ile')
    ax_sp.fill_between(ages, data['spend_p25'], data['spend_p75'],
                        alpha=0.3, label='25th-75th %ile')
    ax_sp.plot(ages, data['spend_median'], linewidth=2, label='Median')
    ax_sp.axvline(x=opt_retire, color='r', linestyle='--', alpha=0.5)
    ax_sp.axhline(y=config.spending_floor, color='gray', linestyle=':', alpha=0.5,
                   label=f'Floor ${config.spending_floor:.0f}k')
    ax_sp.set_xlabel('Age')
    ax_sp.legend(fontsize=7)
    if col == 0:
        ax_sp.set_ylabel('Annual Spending ($k/yr)')

fig.suptitle(f'Monte Carlo: Retire@{opt_retire}, Vitality-Weighted Spending', fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(ASSETS, '4_monte_carlo.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

print("\n  === Monte Carlo Summary ===")
print(f"  {'Portfolio':<28s}  {'E[U]':>10s}  {'CE':>14s}")
print(f"  {'-'*56}")
for name, data in mc_results.items():
    print(f"  {name:<28s}  {fmt_u(data['mean_utility']):>10s}  "
          f"({format_money(data['mean_ce'])}/yr)")


# -----------------------------------------------------------------------
# 5. Stress test: Bayesian + stochastic lifespan/income
# -----------------------------------------------------------------------
print(f"\n[5/7] Stress test: + stochastic lifespan (300 sims)...")

stress_config = replace(mc_config,
                        stochastic_lifespan=True)
stress_scorer = CRRAUtility.from_config(stress_config)
stress_results = run_monte_carlo(stress_config, mc_spending, stress_scorer, n_simulations=300)

stress_show = {k: v for k, v in stress_results.items() if k != 'CashPortfolio'}
n_stress_show = len(stress_show)
fig, axes = plt.subplots(2, n_stress_show, figsize=(6 * n_stress_show, 9))
if n_stress_show == 1:
    axes = axes.reshape(2, 1)

for col, (name, data) in enumerate(stress_show.items()):
    ages = data['ages']
    u = data.get('mean_utility', 0)
    ce = data.get('mean_ce', 0)

    # Top row: Net worth
    ax_nw = axes[0, col]
    ax_nw.fill_between(ages, data['p10'], data['p90'], alpha=0.15, label='10th-90th %ile')
    ax_nw.fill_between(ages, data['p25'], data['p75'], alpha=0.3, label='25th-75th %ile')
    ax_nw.plot(ages, data['median'], linewidth=2, label='Median')
    ax_nw.axvline(x=opt_retire, color='r', linestyle='--', alpha=0.5)
    ax_nw.set_yscale('symlog', linthresh=0.1)
    ax_nw.set_title(f'{name}\nE[U]={fmt_u(u)}  (CE={format_money(ce)}/yr)')
    ax_nw.legend(fontsize=7)
    if col == 0:
        ax_nw.set_ylabel('Net Worth ($M, log scale)')

    # Bottom row: Spending
    ax_sp = axes[1, col]
    ax_sp.fill_between(ages, data['spend_p10'], data['spend_p90'],
                        alpha=0.15, label='10th-90th %ile')
    ax_sp.fill_between(ages, data['spend_p25'], data['spend_p75'],
                        alpha=0.3, label='25th-75th %ile')
    ax_sp.plot(ages, data['spend_median'], linewidth=2, label='Median')
    ax_sp.axvline(x=opt_retire, color='r', linestyle='--', alpha=0.5)
    ax_sp.axhline(y=stress_config.spending_floor, color='gray', linestyle=':',
                   alpha=0.5, label=f'Floor ${stress_config.spending_floor:.0f}k')
    ax_sp.set_xlabel('Age')
    ax_sp.legend(fontsize=7)
    if col == 0:
        ax_sp.set_ylabel('Annual Spending ($k/yr)')

fig.suptitle(f'Stress Test: Retire@{opt_retire} + Stochastic Lifespan (Longevity Risk)',
             fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(ASSETS, '5_stress_test.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

print("\n  === Stress Test Summary ===")
print(f"  {'Portfolio':<28s}  {'E[U]':>10s}  {'CE':>14s}")
print(f"  {'-'*56}")
for name, data in stress_results.items():
    print(f"  {name:<28s}  {fmt_u(data['mean_utility']):>10s}  "
          f"({format_money(data['mean_ce'])}/yr)")


# -----------------------------------------------------------------------
# 6. Vitality curve visualization
# -----------------------------------------------------------------------
print("\n[6/7] Generating vitality curve...")

ages_v = list(range(25, 95))
vitalities = [vitality_at_age(a, config) for a in ages_v]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ages_v, vitalities, linewidth=2.5, color='teal')
ax.fill_between(ages_v, vitalities, alpha=0.2, color='teal')
ax.axhline(y=config.vitality_floor, color='gray', linestyle='--', alpha=0.5,
           label=f'Floor={config.vitality_floor}')
ax.axvline(x=config.vitality_peak_age, color='orange', linestyle='--', alpha=0.5,
           label=f'Peak age={config.vitality_peak_age}')
ax.set_xlabel('Age')
ax.set_ylabel('Vitality Multiplier')
ax.set_title(f'Vitality Curve: v(age) = {config.vitality_floor} + '
             f'{1-config.vitality_floor:.1f} * exp(-((age-{config.vitality_peak_age})/'
             f'{config.vitality_half_life:.0f})^2)\n'
             f'Drives spending front-loading: spend more at 35, less at 80')
ax.set_ylim(0, 1.1)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(ASSETS, '6_vitality_curve.png'), dpi=150, bbox_inches='tight')
plt.close(fig)


# -----------------------------------------------------------------------
# 7. PDF comparison of lifetime E[U] across leverage levels
# -----------------------------------------------------------------------
print("\n[7/7] Running E[U] distributions across leverage levels (300 sims each)...")

pdf_leverages = [1.0, 1.5, 2.0, 3.0, 4.0]
pdf_spending = VitalityAmortizedSpending()
pdf_seeds = _generate_seeds(300)

leverage_utilities = {}
for lev in pdf_leverages:
    pdf_cfg = replace(mc_config, leverage_ratio=lev)
    pdf_scorer = CRRAUtility.from_config(pdf_cfg)
    batch = _run_sim_batch(pdf_cfg, pdf_spending, pdf_seeds)
    utils = []
    for result, result2 in batch:
        u = _score_utility(pdf_scorer, result, result2, 2, pdf_cfg)
        utils.append(u)
    leverage_utilities[lev] = np.array(utils)
    mean_u = float(np.mean(utils))
    print(f"  {lev:.1f}x:  E[U]={mean_u:,.1f}  "
          f"med={float(np.median(utils)):,.1f}  "
          f"std={float(np.std(utils)):,.1f}")

# Cap at global P95 across all leverage levels
all_utils = np.concatenate(list(leverage_utilities.values()))
U_CAP = float(np.percentile(all_utils, 95))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(pdf_leverages)))

for (lev, utils), color in zip(leverage_utilities.items(), colors):
    sorted_u = np.sort(utils)
    cdf = np.arange(1, len(sorted_u) + 1) / len(sorted_u)
    pct_above = 100 * float(np.mean(utils > U_CAP))
    lbl = f'{lev:.1f}x'
    if pct_above > 0:
        lbl += f' ({pct_above:.0f}%>{U_CAP:.0f})'
    mask = sorted_u <= U_CAP
    ax1.plot(sorted_u[mask], cdf[mask], linewidth=2, color=color, label=lbl)

ax1.set_xlim(right=U_CAP)
ax1.set_xlabel('Lifetime E[U]')
ax1.set_ylabel('Cumulative Probability')
ax1.set_title(f'CDF of Lifetime Utility by Leverage (capped at P95={U_CAP:.0f})')
ax1.legend(fontsize=8, title='Leverage')
ax1.grid(True, alpha=0.3)

# Right panel: box plot, capped
box_data = [np.clip(leverage_utilities[lev], None, U_CAP) for lev in pdf_leverages]
above_counts = [int(np.sum(leverage_utilities[lev] > U_CAP)) for lev in pdf_leverages]
bp = ax2.boxplot(box_data, tick_labels=[f'{lev:.1f}x' for lev in pdf_leverages],
                  patch_artist=True, showmeans=True,
                  meanprops=dict(marker='D', markerfacecolor='red', markersize=6))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)
for i, (lev, n_above) in enumerate(zip(pdf_leverages, above_counts)):
    if n_above > 0:
        ax2.text(i + 1, U_CAP * 0.97, f'+{n_above} above',
                 ha='center', va='top', fontsize=7, color='red')
ax2.set_ylim(top=U_CAP)
ax2.set_xlabel('Leverage')
ax2.set_ylabel('Lifetime E[U]')
ax2.set_title(f'Utility Distribution (capped at P95={U_CAP:.0f})')
ax2.grid(True, alpha=0.3, axis='y')

fig.suptitle(f'Lifetime Utility PDF by Leverage (Retire@{opt_retire}, Leveraged Portfolio)',
             fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(ASSETS, '7_utility_pdf_by_leverage.png'), dpi=150, bbox_inches='tight')
plt.close(fig)


# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
print("\n")
print("=" * 70)
print("  ADVISORY SUMMARY")
print("=" * 70)

# Extract key numbers
lev_data = ret_results.get('LeveragedStockPortfolio', {})
stock_data = ret_results.get('StockPortfolio', {})
cash_data = ret_results.get('CashPortfolio', {})

print(f"""
  YOUR PROFILE
    Age: {config.start_age}  |  Income: $350-450k  |  Current NW: ${config.initial_net_worth}k
    Expenses: ${config.initial_expenses}k/yr  |  Spending floor: ${config.spending_floor}k/yr
    Utility: U = sum[ B^t * vitality(age) * fire_mult * c^{config.utility_power} ]

  OPTIMAL RETIREMENT AGES (utility-maximizing)
    Cash portfolio:       retire @ {cash_data.get('optimal_age','?')}  ->  E[U]={fmt_u(cash_data.get('optimal_utility',0))}  (CE={format_money(cash_data.get('optimal_ce',0))}/yr)
    Stock portfolio:      retire @ {stock_data.get('optimal_age','?')}  ->  E[U]={fmt_u(stock_data.get('optimal_utility',0))}  (CE={format_money(stock_data.get('optimal_ce',0))}/yr)
    Leveraged portfolio:  retire @ {lev_data.get('optimal_age','?')}  ->  E[U]={fmt_u(lev_data.get('optimal_utility',0))}  (CE={format_money(lev_data.get('optimal_ce',0))}/yr)

  LEVERAGE
    Kelly optimal: {kelly:.2f}x  |  Half-Kelly: {kelly/2:.2f}x
    Utility-maximizing: {best_lev:.2f}x  E[U]={fmt_u(best_u)}  (CE={format_money(best_ce)}/yr)  ruin={best_ruin:.1%}

  JOINT OPTIMUM (Adam optimizer)
    Retire @ {int(round(final_ret))}, leverage {final_lev:.1f}x  ->  E[U]={fmt_u(final_u)}

  MONTE CARLO @ RETIRE@{opt_retire} (Leveraged Portfolio)
    {'':24s}  {'E[U]':>10s}  {'CE':>14s}
    {'-'*52}""")
pf_name = 'LeveragedStockPortfolio'
d_mc = mc_results[pf_name]
d_st = stress_results[pf_name]
print(f"    Base:                {fmt_u(d_mc['mean_utility']):>10s}  "
      f"({format_money(d_mc['mean_ce'])}/yr)")
print(f"    Stress:              {fmt_u(d_st['mean_utility']):>10s}  "
      f"({format_money(d_st['mean_ce'])}/yr)")

print(f"""
  KEY ASSUMPTIONS
    FIRE multiplier: {config.fire_multiplier}x (retirement years worth {config.fire_multiplier}x working years)
    Vitality: peaks at {config.vitality_peak_age}, floor {config.vitality_floor} (front-loads spending)
    SS: claiming @ {config.ss_claiming_age}  |  Tax advantage: {config.retirement_tax_advantage:.0%}
    Market: ERP={config.equity_risk_premium:.1%}, vol={config.stock_vol:.0%}, bond yield={config.initial_bond_yield:.0%}

  CHARTS SAVED TO assets/
    1_retirement_sweep.png       — E[U] vs retirement age per portfolio
    2_leverage_sweep.png         — NW, ruin risk, and E[U] vs leverage
    3_2d_retirement_leverage.png — contour: E[U] over retirement age x leverage
    4_monte_carlo.png            — Monte Carlo at retire@{opt_retire} (all portfolios)
    5_stress_test.png            — Stress test at retire@{opt_retire} (+ stochastic lifespan)
    6_vitality_curve.png         — the vitality function driving spending allocation
    7_utility_pdf_by_leverage.png — PDF of lifetime E[U] across leverage levels
""")
