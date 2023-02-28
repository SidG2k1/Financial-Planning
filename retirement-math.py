# All numbers in present dollar terms, 1 unit = $1000 (2023 USD)

from numpy import random
import numpy as np
from typing import List

seed = np.random.randint(0, 1000000)
year = 2023

def real_mkt_return():
    """
    This function will return the real market return
    """
    # Assume 3.5% real return with 10% volatility
    # use seed + year to make sure the same return is used for the same year
    np.random.seed(seed + year)
    return 1 + np.random.normal(0.035, 0.1)

class Portfolio():
    """
    This is a parent class for all portfolios
    """
    def __init__(self):
        pass

    def pass_year(self):
        """
        This function will pass a year and make decisions
        """
        pass

    def get_nw(self):
        """
        This function will return the net worth of the portfolio
        """
        pass

    def get_nw_history(self):
        """
        This function will return the net worth history of the portfolio
        """
        pass

    def remove_money(self, amount):
        """
        This function will remove money from the portfolio
        """
        pass

    def add_money(self, amount):
        """
        This function will add money to the portfolio
        """
        pass

class PortfolioManager():
    """
    This class will manage the portfolios and make decisions
    """
    def __init__(self, portfolios: List[Portfolio]):
        self.portfolios = portfolios
    
    def pass_year(self):
        """
        This function will pass a year and make decisions
        """
        for portfolio in self.portfolios:
            portfolio.pass_year()
    
    def get_nw(self):
        """
        This function will return the net worth of all the portfolios
        """
        # Return the class name and the net worth
        return [(type(portfolio).__name__, portfolio.get_nw()) for portfolio in self.portfolios]
    
    def get_nw_history(self):
        """
        This function will return the net worth history of all the portfolios
        """
        return [portfolio.get_nw_history() for portfolio in self.portfolios]
    
    def remove_money(self, amount):
        """
        This function will remove money from the portfolios
        """
        for portfolio in self.portfolios:
            portfolio.remove_money(amount)

    def add_money(self, amount):
        """
        This function will add money to the portfolios
        """
        for portfolio in self.portfolios:
            portfolio.add_money(amount)

class CashPortfolio(Portfolio):
    """
    This class will represent a cash portfolio
    """
    def __init__(self, init_nw):
        self.nw = init_nw
        self.nw_history = [init_nw]
    
    def pass_year(self):
        """
        This function will pass a year and make decisions
        """
        self.nw /= 1.02 # Assume 2% inflation
        self.nw_history.append(self.nw)
    
    def get_nw(self):
        """
        This function will return the net worth of the portfolio
        """
        return self.nw
    
    def get_nw_history(self):
        """
        This function will return the net worth history of the portfolio
        """
        return self.nw_history
    
    def remove_money(self, amount):
        """
        This function will remove money from the portfolio
        """
        self.nw -= amount

    def add_money(self, amount):
        """
        This function will add money to the portfolio
        """
        self.nw += amount

class StockPortfolio(Portfolio):
    """
    This class will represent a stock portfolio
    """
    def __init__(self, init_nw):
        self.nw = init_nw
        self.nw_history = [init_nw]

    def pass_year(self):
        """
        This function will pass a year and make decisions
        """
        self.nw *= real_mkt_return()
        self.nw_history.append(self.nw)
    
    def get_nw(self):
        """
        This function will return the net worth of the portfolio
        """
        return self.nw

    def get_nw_history(self):
        """
        This function will return the net worth history of the portfolio
        """
        return self.nw_history

    def remove_money(self, amount):
        """
        This function will remove money from the portfolio
        """
        self.nw -= amount
    
    def add_money(self, amount):
        """
        This function will add money to the portfolio
        """
        self.nw += amount

class LeveragedStockPortfolio(Portfolio):
    """
    This class will represent a leveraged stock portfolio
    """
    def __init__(self, init_nw, init_leverage):
        self.nw = init_nw
        self.nw_history = [init_nw]
        self.leverage = init_leverage

    def pass_year(self):
        """
        This function will pass a year and make decisions
        """
        fees = (self.leverage - 1) * 0.02
        self.nw *= ((real_mkt_return() - 1) * self.leverage + 1) - fees
        self.nw_history.append(self.nw)

    def get_nw(self):
        """
        This function will return the net worth of the portfolio
        """
        return self.nw

    def get_nw_history(self):
        """
        This function will return the net worth history of the portfolio
        """
        return self.nw_history

    def remove_money(self, amount):
        """
        This function will remove money from the portfolio
        """
        self.nw -= amount

    def add_money(self, amount):
        """
        This function will add money to the portfolio
        """
        self.nw += amount

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

portfolios = PortfolioManager([
    CashPortfolio(nw),
    StockPortfolio(nw),
    LeveragedStockPortfolio(nw, init_leverage=2),
])

one_time_expenses = {
    28: 50, # Marriage
    32: 300, # Down payment on house
}
for i in range(25, 70, 5): one_time_expenses[i] = 40 # Car

for curr_age in range(age, expected_lifespan + 1):
    # Since all math is done in current dollar terms, we need to adjust for inflation
    inc = post_tax(realized_income[curr_age - age])
    exp = expenses[curr_age - age]
    if curr_age in one_time_expenses: exp += one_time_expenses[curr_age]

    # Update the portfolio
    portfolios.pass_year()
    portfolios.remove_money(exp)
    portfolios.add_money(inc)

    print('age: ', curr_age, end=';\t')
    # format the numbers
    print('post-tax income: ', format_money(inc), end=';\t')
    print('expenses: ', format_money(exp), end=';\t')
    print('nw: ', portfolios.get_nw())
    year += 1

import matplotlib.pyplot as plt
# Plot the net worth history of each portfolio over time (in millions)
#   Use the legend to label each portfolio using the class name
nw_history = [portfolio.get_nw_history() for portfolio in portfolios.portfolios]
# normalize the net worth history (shift up index by age): turn to dict
nw_history = {i + age - 2: [nw_history[j][i] / 1000 for j in range(len(nw_history))] for i in range(len(nw_history[0]))}
# plot the net worth history
plt.plot(list(nw_history.keys()), list(nw_history.values()))
# Add a legend
plt.legend([type(portfolio).__name__ for portfolio in portfolios.portfolios])
# Add labels to the axes
plt.xlabel('Age')
plt.ylabel('Net Worth ($M)')
# Show fat red line at retirement age and 0 net worth
plt.axvline(x=ret_age, color='r', linestyle='--')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()
