# Model notes

## Workbook compatibility

The deterministic engine follows the workbook’s economic balance-sheet identity:

`financial wealth + human capital − consumption liabilities − insurance liability = net worth`

It reproduces the workbook’s Gompertz calibration, salary coefficients, reference portfolio,
covariance matrix, equilibrium discount rates, consumption divisor, annuity credits, insurance
pricing, and financial-allocation transformation.

The default bequest solver is a continuous bounded optimizer over the composite utility model.
The workbook evaluates a 1%-spaced grid, and its displayed utility saturates at floating-point
precision for large dollar values. Consequently, the compatibility scenario’s fixed workbook
bequest reproduces the workbook balance sheet exactly, while the decision-native optimum can
differ. This is intentional and covered by parity tests.

The workbook treats nondiscretionary consumption as a hard liability. That input defaults to zero
in decision-native scenarios. The default $40,000 spending preference is instead a bounded
`SpendingFloorCurve`, allowing its importance to be compared with every other desire. The
compatibility scenario retains the workbook liability explicitly.

`IsoelasticCurve` preserves ordinary CRRA utility at and above its dollar reference. Below the
reference, a smooth monotone transform approaches the configured finite lower bound without a
flat segment, so low consumption and bequest choices remain distinguishable.

Retirement and work quality are explicit annual status flows, not multipliers on consumption
utility. `RETIRED` and `WORKING` add-ons can use `LinearCurve`, discounted-sum aggregation, and an
exponential age profile. The legacy `retirement_utility_multiplier` is accepted only at its neutral
value of one. Allocation risk tolerance defaults to the CRRA consumption elasticity; an explicit
override is required to separate them.

`WorkbookSocialInsurance` reproduces the spreadsheet estimate. `SocialSecurityPolicy` provides a
separate 2026 real-dollar estimate using the published $1,286 and $7,749 monthly PIA bend points
and the $184,500 taxable maximum. It does not replace an SSA earnings-record estimate.

Policy sources: [SSA PIA bend points](https://www.ssa.gov/oact/COLA/bendpoints.html) and
[SSA 2026 contribution base](https://www.ssa.gov/oact/COLA/cbbdet.html).

## Stochastic engine

The default Monte Carlo return model is `RegimeSwitchingMarket`, a three-state multivariate
Student-t model for equity, bonds, and cash. Normal, growth/deflation-stress, and
inflation-stress states have distinct moments and cross-asset covariance. Bonds tend to diversify
equity in growth stress and move with equity in inflation stress. Persistent regimes, stochastic
volatility, and mean-reverting rates create serial dependence without directly autoregressing
returns. `StochasticMarket` remains the legacy compatibility model selected with
`--market-model legacy`.

The log-volatility state starts in its stationary distribution. Its multiplier is normalized by
its stationary second moment, so configured volatility remains the root-mean-square innovation
scale rather than increasing with volatility-of-volatility or simulation age.

Equity gross returns use a symmetric bounded transform of a latent log-return shock. This
retains the fitted conditional mean while enforcing limited liability. Bond duration applies to
unexpected rate innovations; predictable mean reversion is part of the fitted term premium rather
than a mechanical bond windfall. The linear duration approximation is floored only when its
extreme tail would otherwise imply an impossible loss of 100% or more, so both unlevered assets
respect limited liability.

The checked-in defaults are an offline fit to annual US real returns through 2025. The runtime
imports generated constants and performs no data access or estimation. Regimes are classified
from lower-quartile equity log excess returns, split by the sign of bond excess returns.
Transition frequencies, conditional moments, Student-t tails, and real-rate dynamics are fitted
from that sample.

Each simulated path draws one deterministic-bootstrap parameter scenario and retains it across
its lifetime. This separates uncertainty about fitted premiums, volatilities, and transition
probabilities from annual regime and innovation risk. The ensemble is centered on the fitted
unconditional moments; it is estimation uncertainty, not a current-market forecast. These
fitted defaults belong to `RegimeSwitchingMarket`; the legacy generator and deterministic
workbook model retain their compatibility assumptions.

Retained epistemic ensembles declare the transition and initial-probability centers they were
fitted around. Changing either center requires dropping the ensemble or supplying a matching one.
Global rate-volatility overrides scale the fitted regime values; tail degrees and stock-bond
correlations are regime-specific and rejected as global overrides.

`conditioned_on_epistemic_scenario()` instead fixes one parameter scenario across every path in a
generation call. This supports nested Monte Carlo: outer worlds represent uncertainty about the
return model, while inner paths vary only regimes and innovations conditional on that model.

Equity premium and volatility sweeps remain unconditional: premium changes shift both regime
premiums from their defaults, while volatility changes scale both conditional regime shocks.

The Monte Carlo engine uses vectorized paths and:

- correlated Student-t innovations normalized to unit variance;
- stochastic log-volatility;
- mean-reverting real rates;
- a duration approximation for bond returns;
- an `IncomeRiskModel` protocol, defaulting to `TransitoryMarketJobLoss` for market-correlated job loss;
- conditional Gompertz death ages;
- generic, futures, and box-spread leverage;
- explicit limited liability and margin-call cooldowns.

Insolvency means both available financial wealth and current income are exhausted while alive.
Policy shortfall means available cash could not fund the optimizer’s selected spending. A
preference breach records how often an outcome enters a curve’s diagnostic region and its utility
cost; it does not change either physical diagnostic.

The rolling optimizer may purchase additional permanent insurance using current cash. Benefits
arrive at simulated death; coverage that survives the modeled horizon is credited at its terminal
boundary. Total utility, its named components, rolling controls, and preference diagnostics are
retained per path. Insured bequest is a decision path; terminal bequest is the resulting estate.
Future income informs continuation utility but cannot fund the current spending or insurance
control. Leverage selection considers the exact configured cap and the chosen instrument’s
financing, roll, dividend-tax, and expected futures-tax drag.

## Taxes and accounts

The tax policy is an annual planning estimate. It handles ordinary brackets, the standard
deduction, preferential-gain stacking, taxable Social Security, NIIT, state tax, and payroll tax.
It intentionally omits credits, AMT, itemization, RMDs, IRMAA, ACA subsidies, and jurisdictional
rules unless a caller supplies a replacement policy.

The bundled 2026 brackets, standard deduction, and preferential-gain thresholds come from
[IRS Revenue Procedure 2025-32](https://www.irs.gov/irb/2025-45_IRB).

Account projections report tax ledgers separately from portfolio market value. Applications can
schedule withdrawals to pay those taxes from a selected account.

Sales and asset-location rebalances require an explicit model year for gain classification.
Withdrawals and Roth conversions return requested, executed, and shortfall amounts; multi-account
mutations validate or stage all work before committing it.
