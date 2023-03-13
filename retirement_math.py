# All numbers in present dollar terms, 1 unit = $1000 (2023 USD)

import matplotlib.pyplot as plt
from time import time as curr_time
from portfolios import *

def format_money(val):
    # * 1000 to convert to $1000 units, add commas, and add $
    return '$' + '{:,}'.format(int(val * 1000))

mode = "plot"
mode = "stat"
simulations = 0
if mode == 'stat':
    target_run_dur_secs = 3
    simulations = target_run_dur_secs * 10000 // 20
else:
    simulations = 5

age = 22
ret_age = 60
expected_lifespan = 100
lifestyle_inflation = 1.01
nw = 50

mean_case = [230] * 3 + [260] * 3 + [400] * 4 + [550] * 200
# optimistic_case = [450] + [350] * 4 + [450] * 3 + [600] * 6 + [800] * 100
income = mean_case

realized_income = income[:1 + ret_age - age] + \
    [0] * (expected_lifespan - ret_age)

init_exp = 60
expenses = [init_exp * lifestyle_inflation**i for i in range(200)]

expenses = [60] * 10 + [60 + 30] * 20 + [60] * 10 + [40] * 100
#           new grad,   child,          post-child,  retirement

start_time = curr_time()

def run_sim():
    reset_seed()
    set_year(2023)

    high_risk_portfolio = LeveragedStockPortfolio(0)
    med_high_risk_portfolio = HybridPortfolio(
        0, [LeveragedStockPortfolio(0), StockPortfolio(0)], [0.5, 0.5])
    medium_risk_portfolio = StockPortfolio(0)
    low_risk_portfolio = CashPortfolio(0)

    transition_portfolios = TemporalHybridPortfolio(0, [LeveragedStockPortfolio(0), 
                                                        HybridPortfolio(0, [LeveragedStockPortfolio(0), StockPortfolio(0)], [0.5, 0.5]), 
                                                        StockPortfolio(0), 
                                                        CashPortfolio(0)], [
                                                    year - age + 0 * expected_lifespan/4,
                                                    year - age + 1 * expected_lifespan/4,
                                                    year - age + 2 * expected_lifespan/4,
                                                    year - age + 3 * expected_lifespan/4,
                                                    year - age + 4 * expected_lifespan/4,
                                                    ])

    portfolios = PortfolioManager([
        high_risk_portfolio,
        med_high_risk_portfolio,
        medium_risk_portfolio,
        low_risk_portfolio,
        transition_portfolios, # Note this has a bug
    ])

    one_time_expenses = {
        28: 50,  # Marriage
    }
    for i in range(25, 70, 5):
        one_time_expenses[i] = 40  # Car

    total_history = []
    for p in portfolios.portfolios: p.add_money(nw)
    for curr_age in range(age, expected_lifespan + 1):
        # Since all math is done in current dollar terms, we need to adjust for inflation
        inc = post_income_tax(realized_income[curr_age - age])
        exp = expenses[curr_age - age]
        if curr_age in one_time_expenses:
            exp += one_time_expenses[curr_age]

        # Update the portfolio
        portfolios.pass_year()
        portfolios.remove_money(exp)
        portfolios.add_money(inc)

        total_history.append(portfolios.get_nw())
        incr_yr()
    return total_history, portfolios

import matplotlib.pyplot as plt

all_runs = [] # list of clarified data

if mode == 'plot':
    simulations = 5

for i in range(simulations):
    total_history, portfolios = run_sim()

    # We want {"port_type" : ([age, ...], [nw, ...]), ...}
    clarified = {}
    for port_idx, port_name in zip(range(len(total_history[0])), [type(portfolio).__name__ for portfolio in portfolios.portfolios]):
        ages = [curr_age for curr_age in range(age, expected_lifespan + 1)]
        nw_at_age_idx = [total_history[curr_age - age][port_idx][1] / 1000 for curr_age in range(age, expected_lifespan + 1)]
        clarified[port_name] = (ages, nw_at_age_idx)

    if mode == 'stat':
        # aggregate the history and portfolio data
        all_runs.append(clarified)
    else:
        legend = []
        for port_name, nw_data in clarified.items():
            plt.plot(nw_data[0], nw_data[1])
            legend.append(port_name)
        plt.legend(legend)

        # Add labels to the axes
        plt.xlabel('Age')
        plt.ylabel('Net Worth ($M)')
        # Show fat red line at retirement age and 0 net worth
        plt.axvline(x=ret_age, color='r', linestyle='--')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.show()

if mode == 'stat':
    # Comprehend all_runs = [{"port_type" : ([age, ...], [nw, ...]), ...}, ...]

    # {"port_type": [nw1, ...], ...}
    type_to_end_nw = {}
    for run in all_runs:
        for port_name, nw_data in run.items():
            if port_name not in type_to_end_nw:
                type_to_end_nw[port_name] = []
            type_to_end_nw[port_name].append(int(nw_data[1][-1] * 10)/10) # Round to nearest 0.1M
    
    # We want average, percent negative, and ideally cumalative density function (essentiall P{0 --> 100})

    """
    type:
        metric1: data
        metric2: data
    type2:
        ...
    """

    # TODO: Consider head-to-heads (how often does portfolio X beat Y)

    # {"type": {"metric": data}}
    type_to_agg = {}
    for type, nw_traj in type_to_end_nw.items():
        if type not in type_to_agg:
            type_to_agg[type] = {}

        # Pct neg
        pct_neg = len(list(filter(lambda x: x < 0, nw_traj))) / len(nw_traj)
        type_to_agg[type]["Success (Positive) Rate"] = str(100 - int(pct_neg * 100)) + "%"

        # Average
        type_to_agg[type]["Average Ending NW"] = str(int(10 * sum(nw_traj) / len(nw_traj)) / 10) + 'M'

        """
        for port_name, nw_data in clarified.items():
            plt.plot(nw_data[0], nw_data[1])
            legend.append(port_name)
        plt.legend(legend)

        # Add labels to the axes
        plt.xlabel('Age')
        plt.ylabel('Net Worth ($M)')
        # Show fat red line at retirement age and 0 net worth
        plt.axvline(x=ret_age, color='r', linestyle='--')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.show()
        """
        import numpy as np
        pctiles = [i for i in range(0, 99)]
        plt.plot(pctiles, np.percentile(nw_traj, pctiles))
        # plt.plot(sorted(nw_traj))
        plt.legend([type])
        plt.xlabel("Percentile")
        plt.ylabel("NW")
        plt.axhline(y=0, color='r', linestyle='--')
        plt.show()


        # next agg
        continue
    
    duration = int(curr_time() - start_time)
    print('Based on', simulations, 'simulations, taking', duration, 'seconds we get the following results:')
    print('===========================')
    for type, info in type_to_agg.items():
        print(type, ':')
        for metric, data in info.items():
            print('\t', metric, ':', '\t', data)
    
