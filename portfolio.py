
from typing import List
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
    def __init__(self, init_nw, real_mkt_return_func):
        self.nw = init_nw
        self.nw_history = [init_nw]
        self.real_mkt_return_func = real_mkt_return_func

    def pass_year(self):
        """
        This function will pass a year and make decisions
        """
        self.nw *= self.real_mkt_return_func()
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
    def __init__(self, init_nw, init_leverage, real_mkt_return_func):
        self.nw = init_nw
        self.nw_history = [init_nw]
        self.leverage = init_leverage
        self.real_mkt_return_func = real_mkt_return_func

    def pass_year(self):
        """
        This function will pass a year and make decisions
        """
        fees = (self.leverage - 1) * 0.02
        self.nw *= ((self.real_mkt_return_func() - 1) * self.leverage + 1) - fees
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