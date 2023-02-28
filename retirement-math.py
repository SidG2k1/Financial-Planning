# All numbers in present dollar terms, 1 unit = $1000 (2023 USD)

def real_mkt_return():
    import numpy as np
    return 1 + np.random.normal(0.035, 0.1)


def product(iterable):
    res = 1
    for i in iterable:
        res *= i
    return res

def post_tax(income):
    income *= 1000
    # from key to next key is the tax bracket
    # Assume flat 15% state tax
    state_tax = 0.15
    tax_map = {
        0: 0,
        9950: 0.1,
        40525: 0.12,
        86375: 0.22,
        164925: 0.24,
        209425: 0.32,
        523600: 0.35,
        10**10: 0.37
    }
    tax_map = list(tax_map.items())
    tax_map.sort(key=lambda x: x[0])

    income_backup = income
    total_tax = 0
    for i in range(len(tax_map) - 1):
        # if the income is less than the next bracket, then we're done
        bracket_size = tax_map[i + 1][0] - tax_map[i][0]
        if income <= bracket_size:
            total_tax += income * tax_map[i][1]
            break
        # otherwise, we need to pay the tax on the entire bracket
        total_tax += bracket_size * tax_map[i][1]
        income -= bracket_size
    return (income_backup - total_tax - income_backup * state_tax) / 1000

def format_money(val):
    # * 1000 to convert to $1000 units, add commas, and add $
    return '$' + '{:,}'.format(int(val * 1000))

age = 22
ret_age = 40
expected_lifespan = 100
lifestyle_inflation = 1.01

nw = 40

mean_case = [230] * 3 + [260] * 3 + [400] * 4 + [550] * 200
optimistic_case = [450] * 3 + [550] * 3 + [700] * 5 + [850] * 200
pessimistic_case = [170] * 4 + [200] * 3 + [240] * 5 + [260] * 200
income = mean_case

init_exp = 60
expenses = [init_exp * lifestyle_inflation**i for i in range(200)]
realized_income = income[:1 + ret_age - age] + [0] * (expected_lifespan - ret_age)

retirement_nw = nw
cash_case = nw
age_to_nw = {}
age_to_all_cash_portfolio = {}
for curr_age in range(age, expected_lifespan + 1):
    # Since all math is done in current dollar terms, we need to adjust for inflation
    market_return = real_mkt_return()
    retirement_nw = retirement_nw * market_return
    inc = post_tax(realized_income[curr_age - age])
    retirement_nw += inc
    exp = expenses[curr_age - age]
    retirement_nw -= exp

    cash_case = cash_case + inc - exp
    cash_case /= 1.02 # real return of -2% on cash portfolio due to inflation

    print('age: ', curr_age, end=';\t')
    # format the numbers
    print('market_return: ', '{:.2%}'.format((market_return - 1)), end=';\t')
    print('post-tax income: ', format_money(inc), end=';\t')
    print('expenses: ', format_money(exp), end=';\t')
    print('nw: ', format_money(retirement_nw))

    # These will later be used to plot the net worth over time in millions of dollars
    age_to_nw[curr_age] = retirement_nw / 1000
    age_to_all_cash_portfolio[curr_age] = cash_case / 1000

# exit()
import matplotlib.pyplot as plt
plt.plot(list(age_to_nw.keys()), list(age_to_nw.values()))
plt.plot(list(age_to_all_cash_portfolio.keys()), list(age_to_all_cash_portfolio.values()))
plt.title('Net Worth Over Time')
plt.xlabel('Age')
plt.ylabel('Net Worth (M$)')
plt.grid()
plt.legend(['Investment Portfolio', 'All-Cash Portfolio'])
# Show fat red line at retirement age and 0 net worth
plt.axvline(x=ret_age, color='r', linestyle='--')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()
