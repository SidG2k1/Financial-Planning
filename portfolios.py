import numpy as np
from typing import List
year = 2023

seed = np.random.randint(0, 1000000)

def incr_yr(ct = 1):
    global year
    year += ct

def set_year(yr):
    global year
    year = yr

def reset_seed():
    global seed
    seed = np.random.randint(0, 1000000)

def normal(mean, std):
    """
    Returns a normal distribution with mean and std, with seed + year
    """
    np.random.seed(seed + year)
    res = np.random.normal(0, 1)
    res = res * std + mean
    return res

def global_stocks_return():
    """
    This function will return the real market return
    """
    mean_return = 5
    return normal(1 + mean_return / 100, 0.11)

def global_bonds_return():
    """
    Bonds should have positive yield, yet be negatively correlated to stocks

    Think about if the formulation below makes sense
    """
    stock_return_quality = normal(0, 0.14)
    bond_yield = 1.02 - stock_return_quality / 10
    return max(1.0025, bond_yield)

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
        self.nw /= 1.02  # Assume 2% inflation
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
        if self.nw > 0:
            self.nw *= global_stocks_return()
        else:
            self.nw *= 1.04 # debt interest
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

    def __init__(self, init_nw):
        self.nw = init_nw
        self.nw_history = [init_nw]
        self.leverage = 2

    def pass_year(self):
        """
        This function will pass a year and make decisions
        """
        # 2% fee per leverage
        fees = (self.leverage - 1) * 0.02
        if self.nw > 0:
            self.nw *= ((global_stocks_return() - 1) * self.leverage + 1) - fees
        else:
            self.nw *= 1.04 # debt interest
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


class HybridPortfolio(Portfolio):
    """
    This class will represent a hybrid portfolio.
    Given an array of portfolios, and an array of weights, it will
    manage the portfolios rebalance them each year
    Note: weights must sum to <= 1 (with remaining money automatically in cash)
    """

    def __init__(self, init_nw, portfolios: List[Portfolio], weights: List[float]):
        # Create new portfolios of the same class, just reset the net worth to be weight * init_nw
        self.nw = init_nw
        self.nw_history = [init_nw]
        self.weights = weights
        tot_weight = 0
        for weight in weights:
            if weight < 0:
                raise ValueError("Weight cannot be negative")
            tot_weight += weight
        if tot_weight > 1:
            raise ValueError("Sum of weights cannot be greater than 1")
        self.portfolios = []
        for portfolio, weight in zip(portfolios, weights):
            self.portfolios.append(portfolio.__class__(init_nw * weight))
        # If the sum of weights is less than 1, add the rest to cash
        if tot_weight < 1:
            self.portfolios.append(CashPortfolio(init_nw * (1 - tot_weight)))
            self.weights.append(1 - tot_weight)

    def pass_year(self):
        """
        This function will pass a year and make decisions
        """
        # First, pass the year for each portfolio
        for portfolio in self.portfolios:
            portfolio.pass_year()

        # Then, rebalance the portfolios
        # First, get the net worth of each portfolio
        nw = [portfolio.get_nw() for portfolio in self.portfolios]
        # Then, get the total net worth
        total_nw = sum(nw)
        self.nw = total_nw
        # Then, get the total amount of money we want in each portfolio
        target_nw = [weight * self.nw for weight in self.weights]
        # Then, get the amount of money we want to move from each portfolio
        move_nw = [target - current for target, current in zip(target_nw, nw)]
        # Then, move the money
        for portfolio, amount in zip(self.portfolios, move_nw):
            if amount > 0:
                portfolio.add_money(amount)
            else:
                portfolio.remove_money(-amount)

        # Finally, update the net worth
        self.nw = sum([portfolio.get_nw() for portfolio in self.portfolios])
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
        Pull from the first portfolio
        """
        self.portfolios[0].remove_money(amount)

    def add_money(self, amount):
        """
        This function will add money to the portfolio
        Add to the first portfolio
        """
        self.portfolios[0].add_money(amount)


class TemporalHybridPortfolio(Portfolio):
    """
    This class will take in an array of portfolios and an array of years
    It will then switch between the portfolios at the given years
    This is useful for when you want to switch between different portfolios
        usually when pivoting to safer investments as you get older
    """

    def __init__(self, init_nw, portfolios: List[Portfolio], years: List[int]):
        # Sort by years
        portfolios, years = zip(
            *sorted(zip(portfolios, years), key=lambda x: x[1]))
        
        self.portfolios = list(portfolios)
        self.years = list(years)
        self.nw = init_nw
        self.nw_history = [init_nw]
        # add a cash portfolio at the end of time
        self.portfolios.append(CashPortfolio(0))
        self.years.append(1000000000)

        self.current_portfolio = 0

    def pass_year(self):
        """
        This function will pass a year and make decisions
        """
        # If we've passed the year, switch to the next portfolio
        if year >= self.years[self.current_portfolio + 1]:
            # Time to switch!!
            # Move the money from the current portfolio to the next portfolio
            self.portfolios[self.current_portfolio + 1].add_money(
                self.portfolios[self.current_portfolio].get_nw())
            
            self.portfolios[self.current_portfolio].remove_money(
                self.portfolios[self.current_portfolio].get_nw())
            
            self.portfolios[self.current_portfolio].nw = 0
            self.current_portfolio += 1
        # Pass the year for the current portfolio
        self.portfolios[self.current_portfolio].pass_year()
        # Update the net worth
        self.nw = self.portfolios[self.current_portfolio].get_nw()
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
        self.portfolios[self.current_portfolio].remove_money(amount)

    def add_money(self, amount):
        """
        This function will add money to the portfolio
        """
        self.portfolios[self.current_portfolio].add_money(amount)


def post_income_tax(income):
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
