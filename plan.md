# Plan: Marginal Utility-Based Consumption Allocation

## Problem with current approach

`VitalityAmortizedSpending` allocates consumption **linearly proportional to vitality**:
```
c(age) = floor + excess * v(age) / Σ v(age+s) * d^s
```

This is a heuristic. It ignores:
1. **CRRA curvature** — concave utility means the optimal allocation amplifies weight differences nonlinearly
2. **Time preference (β)** — the utility discount rate should pull spending earlier
3. **FIRE multiplier** — retirement utility boost should affect spending allocation, not just scoring

## New approach: Euler equation / marginal utility equalization

For CRRA utility `V = Σ w_t · c_t^α`, the derivative wrt consumption at age t is:

```
∂V/∂c_t = w_t · α · c_t^(α-1)
```

where `w_t = β^t · vitality(age+t) · fire_mult_t`.

**Optimal allocation** equalizes discounted marginal utility across all ages:

```
w_t · α · c_t^(α-1) = λ    for all t
```

Solving: `c_t = (w_t / λ')^γ` where `γ = 1/(1-α)` and λ' absorbs constants.

Budget constraint `Σ c_t · d^t = total_resources` determines λ':

```
K = total_resources / Σ w_t^γ · d^t
c_t = K · w_t^γ
```

For α=0.65: γ ≈ 2.86, so weight differences are amplified ~3x compared to linear vitality weighting. A year with 2x the utility weight gets ~7.3x the consumption.

## Changes

### 1. `spending.py` — Add `MarginalUtilitySpending` class

New `SpendingRule` implementation:

- Compute utility weight for each remaining year: `w_s = β^s · v(age+s) · fire_mult_s`
- Compute CRRA exponent: `γ = 1 / (1 - α)` where α = `config.utility_power`
- Compute raised weights: `w_s^γ` for each remaining year
- Budget-constrained allocation: `K = excess / Σ w_s^γ · d^s`
- Current year's consumption: `floor + K · w_0^γ`
- Spending floor handling: reserve `PV(floor)` first, then allocate excess via MU-optimal weights (same pattern as VitalityAmortizedSpending)
- Degeneracy handling: if all weights are ~0 or α ≈ 1, fall back to flat annuity

Keep `VitalityAmortizedSpending` in the codebase for backward compatibility.

### 2. `sweeps.py` — Update default spending rule

In `run_retirement_sweep()` and `run_2d_sweep()`: when `vitality_floor < 1.0`, default to `MarginalUtilitySpending()` instead of `VitalityAmortizedSpending()`.

### 3. `retirement_math.py` — Update CLI wiring

- Import `MarginalUtilitySpending`
- Use it as default when `--amortized` or sweep modes are active (replacing `VitalityAmortizedSpending` in the same conditions)

### 4. `CLAUDE.md` — Update documentation

- Update spending rule description to describe the new MU-optimal approach
- Update the "Three Layers" Decision Model section

### 5. `README.md` — Update if it exists

Update any references to spending rules.

## Key design decisions

- **Closed-form over iterative**: For separable CRRA, the Euler equation has an analytical solution. This is mathematically equivalent to iterating MU equalization to convergence but faster. The MU derivative is explicitly computed in the weight-raising step (`w^γ` encodes the MU-optimal response).
- **No new config parameters**: Uses existing `utility_power`, `discount_rate`, `fire_multiplier`, and vitality parameters already in `SimulationConfig`.
- **No circular imports**: The spending rule reads utility parameters from `SimulationConfig` and computes weights directly (same formula as `CRRAUtility._weight()`), avoiding importing from `utility.py`.
