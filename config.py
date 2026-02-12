# Configuration and shared data structures
# All monetary values are in $1,000 units (2023 USD)

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple

if TYPE_CHECKING:
    from portfolio import PortfolioManager


def load_env(env_path: str | None = None) -> Dict:
    """Load personal config overrides from .env file.

    Returns a dict of SimulationConfig field overrides.
    """
    if env_path is None:
        env_path = Path(__file__).parent / '.env'
    else:
        env_path = Path(env_path)

    if not env_path.exists():
        return {}

    raw: Dict[str, str] = {}
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        key, _, val = line.partition('=')
        raw[key.strip()] = val.strip()

    overrides: Dict = {}

    _int_keys = {'RETIREMENT_AGE': 'retirement_age'}
    _float_keys = {
        'INITIAL_NET_WORTH': 'initial_net_worth',
        'INITIAL_EXPENSES': 'initial_expenses',
        'SPENDING_FLOOR': 'spending_floor',
        'STATE_TAX_RATE': 'state_tax_rate',
    }

    for env_key, field_name in _int_keys.items():
        if env_key in raw and raw[env_key]:
            overrides[field_name] = int(raw[env_key])

    for env_key, field_name in _float_keys.items():
        if env_key in raw and raw[env_key]:
            overrides[field_name] = float(raw[env_key])

    if 'INCOME_SCHEDULE' in raw and raw['INCOME_SCHEDULE']:
        overrides['income_schedule'] = [
            float(x.strip()) for x in raw['INCOME_SCHEDULE'].split(',')
            if x.strip()
        ]

    if 'ONE_TIME_EXPENSES' in raw and raw['ONE_TIME_EXPENSES']:
        expenses = {}
        for pair in raw['ONE_TIME_EXPENSES'].split(','):
            pair = pair.strip()
            if ':' in pair:
                age_str, amt_str = pair.split(':', 1)
                expenses[int(age_str.strip())] = float(amt_str.strip())
        overrides['one_time_expenses'] = expenses

    return overrides


@dataclass
class SimulationConfig:
    """All tunable parameters for the retirement simulation.

    Monetary values are in $1,000 units (2023 USD).
    """

    # Personal timeline
    start_age: int = 25
    retirement_age: int = 65
    expected_lifespan: int = 90

    # Starting conditions
    initial_net_worth: float = 50       # $50k
    lifestyle_inflation: float = 1.01   # 1% annual lifestyle creep

    # Income schedule ($1,000 units per year, pre-tax, inflation-adjusted)
    income_schedule: List[float] = field(default_factory=lambda:
        [75] * 5 + [100] * 10 + [120] * 25)

    # Expenses
    initial_expenses: float = 40        # $40k/year
    one_time_expenses: Dict[int, float] = field(default_factory=dict)

    # Market return decomposition: E[stock] = bond_yield + ERP
    equity_risk_premium: float = 0.025  # 2.5% ERP → 4.5% total real equity return
    stock_vol: float = 0.10             # 10% annualized equity volatility (long-run mean)

    # Fat tails: Student-t degrees of freedom for stock return shocks
    stock_tail_df: float = 5.0

    # Volatility clustering: log-normal stochastic volatility
    vol_persistence: float = 0.80       # AR(1) persistence of vol state
    vol_of_vol: float = 0.25            # std of log-vol innovations

    # Stock-bond correlation (discount rate channel)
    stock_bond_corr: float = -0.20

    # Real bond yield dynamics (Vasicek mean-reverting model)
    initial_bond_yield: float = 0.02    # current real yield (~TIPS rate)
    long_run_bond_yield: float = 0.02   # long-run equilibrium theta
    bond_yield_mean_reversion: float = 0.15  # mean reversion speed kappa
    bond_yield_vol: float = 0.012       # annual volatility of yield changes

    # Leverage costs: margin_fee = bond_yield + spread
    margin_spread: float = 0.015        # broker spread above risk-free rate

    # Tax: progressive federal brackets (2023) + flat state rate
    state_tax_rate: float = 0.05
    federal_brackets: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0,        0.00),
        (9_950,    0.10),
        (40_525,   0.12),
        (86_375,   0.22),
        (164_925,  0.24),
        (209_425,  0.32),
        (523_600,  0.35),
        (10**10,   0.37),
    ])

    # Leverage for leveraged portfolio
    leverage_ratio: float = 2.0

    # Utility parameters (used by CRRAUtility, kept here for CLI convenience)
    utility_power: float = 0.65        # CRRA exponent alpha
    discount_rate: float = 0.03        # annual time preference delta (beta = 1/(1+delta))
    fire_multiplier: float = 1.8       # utility mult for retirement years

    # Spending floor: minimum annual spending ($1k). The spending rules reserve
    # resources for this floor first, then front-load only the excess.
    spending_floor: float = 30          # $30k/yr minimum

    # Social Security
    ss_enabled: bool = True
    ss_fra: int = 67                    # Full Retirement Age
    ss_claiming_age: int = 67           # When to start claiming (62-70)
    ss_taxable_max: float = 169         # $169k SS taxable earnings cap ($1k units)

    # Retirement tax advantage: retirement withdrawals are taxed at lower rates
    # (LTCG ~15%) than earned income (~38%). This multiplier inflates the real
    # value of each dollar withdrawn in retirement.
    # Default 1.25 ~ (1-0.15)/(1-0.38) -- a $1 withdrawal buys 25% more than
    # $1 of earned income after tax.
    retirement_tax_advantage: float = 1.25

    # Vitality curve: v(age) = floor + (1-floor)*exp(-((age-peak)/half_life)^2)
    # Captures declining health/energy/capacity to enjoy spending with age.
    # Based on QALY literature: ~1.0 at 30, ~0.80 at 50, ~0.55 at 65, ~0.4 at 80.
    vitality_peak_age: int = 30        # age of peak vitality
    vitality_half_life: float = 35.0   # age offset for ~63% decay of (1-floor)
    vitality_floor: float = 0.3        # minimum vitality (even at 100)

    # Margin call mechanics
    maintenance_margin: float = 0.25   # equity ratio triggering margin call
    margin_call_leverage: float = 1.0  # reduced leverage after margin call

    # MC variance reduction
    antithetic: bool = False           # enable antithetic variates

    # Stochastic lifespan (Gompertz mortality)
    stochastic_lifespan: bool = False
    gompertz_a: float = 0.00003        # baseline hazard
    gompertz_b: float = 0.085          # exponential aging rate
    max_age: int = 110                 # hard upper bound

    # Stochastic income (market-correlated job loss)
    stochastic_income: bool = False
    job_loss_base_prob: float = 0.03   # 3% annual baseline
    job_loss_market_sensitivity: float = 5.0
    job_loss_income_fraction: float = 0.0  # income during job loss (0 = total)

    # Bayesian parameter uncertainty
    bayesian: bool = False
    bayesian_erp_std: float = 0.02
    bayesian_vol_std: float = 0.30     # std of log(vol)
    bayesian_bond_yield_std: float = 0.01

    def __post_init__(self):
        if self.retirement_age < self.start_age:
            raise ValueError(f"retirement_age ({self.retirement_age}) < start_age ({self.start_age})")
        if self.expected_lifespan < self.retirement_age:
            raise ValueError(f"expected_lifespan ({self.expected_lifespan}) < retirement_age ({self.retirement_age})")
        if self.leverage_ratio < 1.0:
            raise ValueError(f"leverage_ratio must be >= 1.0, got {self.leverage_ratio}")
        if self.utility_power <= 0:
            raise ValueError(f"utility_power must be > 0, got {self.utility_power}")
        if self.spending_floor < 0:
            raise ValueError(f"spending_floor must be >= 0, got {self.spending_floor}")


@dataclass
class SimulationResult:
    """Results from a single simulation run."""
    portfolios: PortfolioManager
    spending: List[List[float]]  # spending[portfolio_idx][year_idx], $1k units
    death_age: int | None = None  # actual death age (if stochastic lifespan)


def post_tax(gross_income: float, config: SimulationConfig) -> float:
    """Compute post-tax income given gross income in $1,000 units."""
    gross = gross_income * 1000
    brackets = config.federal_brackets

    federal_tax = 0.0
    remaining = gross
    for i in range(len(brackets) - 1):
        bracket_size = brackets[i + 1][0] - brackets[i][0]
        if remaining <= bracket_size:
            federal_tax += remaining * brackets[i][1]
            break
        federal_tax += bracket_size * brackets[i][1]
        remaining -= bracket_size

    state_tax = gross * config.state_tax_rate
    return (gross - federal_tax - state_tax) / 1000


def format_money(val: float) -> str:
    """Format a $1,000-unit value as a human-readable dollar string."""
    return '${:,}'.format(int(val * 1000))
