import numpy as np
from pqr.data_preprocessing import set_holding_period
from pqr.portfolios_formation import set_equal_weights
from pqr.portfolios_formation import set_value_weights

def get_benchmark(stock_price:np.array, factor:np.array, universe_return:np.array, equal_weights:bool=True, weighting_factor:np.array=0, 
                  min_threshold:int=np.inf, max_threshold:int=np.inf, max_stocks:int=np.inf, nlargest_max_stocks:bool=True):
    """Calculate benchmark portfolio returns to the previous period;

    Input:

        - stock_price (np.array): matrix with instrument prices in single currency;

        - factor (np.array): matrix with the numerical values of the selected factor. 
        The matrix shape should match with the other loaded matrices (prices for calculating the portfolio yield, weights and filters). 
        The list with binary matrices should be loaded immediately into the functions for determining weights

        - universe_return (np.array): matrix with instrument returns to the previous period

        - equal_weights (bool): If equal_weights==True, the weights in the portfolio will be equal. 
        If False, you will need to load an array with the selected factor (weighting_factor)

        - min_threshold (int): The minimum value of the factor that must be overcome to enter the benchmark portfolio;

        - max_threshold (int): The maximum value of the factor that must not be overcome to enter the benchmark portfolio;

        - max_stocks (int): maximum number of instruments in the benchmark for each period;

        - nlargest_max_stocks (bool): If nlargest_max_stocks==True, the maximum number of instruments will be selected from 
        instruments with highest value of the factor. If False, then from the smallest.

    Output:

        - benchmark_positions_return (np.array): matrix with benchmark returns to the previous period.
    """  
    available_stocks = np.copy(stock_price)
    available_factor = np.copy(factor)
    
    available_stocks = np.nan_to_num(available_stocks)
    available_factor = np.nan_to_num(available_factor)
    
    available_stocks[available_stocks != 0] = 1
    available_factor[available_factor != 0] = 1 
    
    benchmark_base = available_stocks + available_factor
    
    benchmark_base[benchmark_base != 2] = 0
    benchmark_base[benchmark_base == 2] = 1
        
    if min_threshold != np.inf:
            
        looking_min_threshold_factor = np.copy(factor)
        looking_min_threshold_factor[looking_min_threshold_factor < min_threshold] = 0
        looking_min_threshold_factor[looking_min_threshold_factor != 0] = 1
        benchmark_base = benchmark_base * looking_min_threshold_factor
        
    if max_threshold != np.inf:   
            
        looking_max_threshold_factor = np.copy(factor)
        looking_max_threshold_factor[looking_max_threshold_factor > max_threshold] = 0
        looking_max_threshold_factor[looking_max_threshold_factor != 0] = 1
        benchmark_base = benchmark_base * looking_max_threshold_factor
   
    if max_stocks != np.inf:
            
        if nlargest_max_stocks == True:
            n_stocks = np.argsort(benchmark_base)[:,-max_stocks:]
        elif nlargest_max_stocks == False:
            n_stocks = np.argsort(benchmark_base)[:,:max_stocks]

        benchmark_base.fill(0)
        np.put_along_axis(benchmark_base, n_stocks, 1, axis=1)

    if equal_weights==True:
        benchmark_portfolio = set_equal_weights(benchmark_base, benchmark=True)
            
    elif equal_weights==False:

        weighting_factor[weighting_factor < 0] = 0 
        benchmark_portfolio = set_value_weights(benchmark_base, weighting_factor, benchmark=True)

    benchmark_positions_return = benchmark_portfolio * universe_return

    
    return benchmark_positions_return