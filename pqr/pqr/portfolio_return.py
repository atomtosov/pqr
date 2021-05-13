import numpy as np
from numpy import inf
import matplotlib.pyplot as plt
from numpy.core.defchararray import array
import statsmodels.api as sm
from tabulate import tabulate
from pqr.data_preprocessing import set_stock_universe

def get_universe_return(stock_price:np.array, replace_zeros:bool=False):
    """Calculates the yield of the entire sample for each period, based on instrument prices. 
    Used to calculate portfolio and benchmark returns.

    Input:

        - stock_price (np.array): Matrix with instrument prices in single currency;

        - replace_zeros: (bool): Replaces 0 with NaN

    Output:

        - universe_return (np.array): matrix with instrument returns to the previous period
    """    
    if replace_zeros==True:
        stock_price[stock_price == 0] = np.nan
    
    universe_return = stock_price / np.roll(stock_price, np.shape(stock_price)[1]) - 1
    universe_return[universe_return == -inf] = np.nan
    universe_return[universe_return == inf] = np.nan
    universe_return = np.nan_to_num(universe_return)

    return universe_return

def get_portfolio_return(portfolios_lists:list, universe_return:np.array, fee_lists:int=0):
    """Calculates the yield of instruments in the portfolio for each period based on the matrix with weights and the yield of the whole sample

    Input:

        - portfolios_lists (np.array): matrices with portfolio weights;

        - universe_return (np.array): matrix with instrument returns to the previous period

        - fee_lists (int): matrices with portfolios costs (optional)

    Output:

        - position_return_list (np.array): list of matrices with portfolio returns to the previous period
    """       
    position_return_list = []
    
    for i in range(len(portfolios_lists)): 
        
        position_return = portfolios_lists[i]
        position_return *= np.roll(universe_return , -np.shape(universe_return )[1])
        position_return = np.roll(position_return, np.shape(position_return)[1])

        if fee_lists != 0:
            position_return -=  fee_lists[i] 

        position_return_list.append(position_return)

    return position_return_list

def compare_portfolios(quantile_names:list, position_returns:list, benchmark_return:np.array, dates:np.array, daily_tstamp:bool=False):
    """Calculates portfolio returns and financial metrics. Visualizes the results in charts and tables.

    Input:

        - quantile_names (list): list with the names of the portfolios;

        - position_returns (list): list of matrices with portfolio returns to the previous period;

        - benchmark_return (np.array): matrix with benchmark returns to the previous period;

        - dates (np.array): array with dates to sign the axis to the graph;

        - daily_tstamp (bool): If daily_tstamp==True, Sharpe Ratio will be annuallized based on daily data. If False, then based on monthly.

    Output:

        - ratios_list (list): list of ratios
    """     
    alpha_coef = []
    alpha_t = []
    sharpe_ratio = []
    excess_return = []
    mean_return = []
    mean_volatility = []
    cumulative_return = []
    total_return = []
    profit_trades = []
    correlation = []
        
    benchmark_return = np.nansum(benchmark_return, axis=1)
    benchmark_return = np.nan_to_num(benchmark_return)
        
    for i in range(len(quantile_names)):

            portfolio_returns = np.nansum(position_returns[i], axis=1)
            mean_return.append(round(portfolio_returns.mean()*100,2))
            mean_volatility.append(round(portfolio_returns.std().mean()*100,2))
            excess_return.append(round((portfolio_returns.mean() - benchmark_return.mean())*100,2))

            if daily_tstamp==False:
                sharpe_ratio.append(round((portfolio_returns.mean() / portfolio_returns.std().mean())*np.sqrt(12), 2))
            else:
                sharpe_ratio.append(round((portfolio_returns.mean() / portfolio_returns.std().mean())*np.sqrt(252), 2)) 

            x = sm.add_constant(benchmark_return)
            est = sm.OLS(portfolio_returns, x).fit()
            alpha_coef.append(round(est.params[0]*100,2))
            alpha_t.append(round(est.tvalues[0],2))

            benchmark_cumulative_return  = np.cumprod(1+benchmark_return)-1
            cumulative_return.append(np.cumprod(1+portfolio_returns)-1)
            total_return.append(round(100*cumulative_return[i][-1],2))
            
            profit_trades.append(round((portfolio_returns >0).sum() / ((portfolio_returns >0).sum() + (portfolio_returns <=0).sum()),2))
            correlation.append(round(np.corrcoef(portfolio_returns, benchmark_return)[0][1],2))

    cumulative_return.append(benchmark_cumulative_return)
    legend_names = quantile_names.copy()
    legend_names.append('benchmark')

    plt.figure(figsize=(16, 8))  

    for i in range(len(cumulative_return)):
            plt.plot(dates, cumulative_return[i])
            plt.grid()
            plt.legend(legend_names)
            plt.suptitle('Portfolios Cumulative Returns', fontsize=25)
                
    ratios_table = [alpha_coef, alpha_t, sharpe_ratio, mean_return, excess_return, total_return, profit_trades, correlation]

    print(tabulate(ratios_table, headers=quantile_names, tablefmt='fancy_grid', showindex=['Alpha %', 'Alpha t', 'SR', 'MR %', 'ER %',
                        'TR %', 'P_trades', 'benchmark_corr']))
        
    ratios_list = [alpha_coef, alpha_t, sharpe_ratio, excess_return, mean_return, mean_volatility, cumulative_return, total_return,
                       profit_trades, correlation]

    return ratios_list