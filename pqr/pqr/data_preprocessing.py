from sys import int_info
import numpy as np
from numpy import inf

def get_factor(factor_data:np.array, static:bool=True, looking_period:int=1, lag:int=0, replace_zeros:bool=False):       
    """Prepares the selected factor for testing:

    - Sets the observation period (makes sense for dynamic factors);

    - Offset by 1 period to prevent lookahead bias;

    - Sets the lag between the receipt of information and the future formation of positions (optional);

    - Calculation of change to previous period (delta) based on the looking period for dynamic factors.
    For example, calculation of momentum from closing prices (optional).

    - Replaces 0 with NaN.  In some cases, null values indicate missing data and it makes sense to replace them with NaN (optional).

    Input:

        - factor_data (np.array): Matrix with the numerical values of the selected factor. 
        The matrix shape should match with the other loaded matrices (prices for calculating the portfolio yield, weights and filters). 
        The list with binary matrices should be loaded immediately into the functions for determining weights

        - static (bool): static=True should be used for data for which it is not necessary to calculate the change. 
        For example, for P/E multipliers, EV/EBITDA, number of messages on Twitter etc.
        static=False counts the ratio to the previous period based on the looking period. 
        For example, for Momentum, changes in trading volume, number of messages on Twitter etc.
        The start of testing will be 1 period later than for static factors.

        - looking_period (int): Use for dynamic factors. Influences the variable n in the formula: t0/(t-n) to calculate factor change. 
        Influences the start period of the test. For static factors, the effect will be equivalent to setting a lag.

        - lag (int): Shifts the final array by n periods. Used to set a delay between the period of data acquisition and portfolio formation.

        - replace_zeros: (bool): Replaces 0 with NaN

    Output:

        - factor (np.array): matrix with processed factors for the subsequent sample cleanup and formation of positions
    """
    if replace_zeros==True:
        factor_data[factor_data == 0] = np.nan    
    
    if static==True:
        factor = np.roll(factor_data, np.shape(factor_data)[1]*looking_period)  
        m = 0
                    
    elif static==False:
        factor = factor_data / np.roll(factor_data, np.shape(factor_data)[1]) -1
        factor = np.roll(factor, np.shape(factor)[1])
        m = 1

    if lag != 0:
        factor = np.roll(factor, np.shape(factor_data)[1]*lag)
        
    factor[:looking_period+lag+m] = np.nan
    factor[factor == -inf] = np.nan
    factor[factor == inf] = np.nan
    
    return factor

def set_stock_universe(factor:np.array, filter_factor:np.array, min_threshold:int=-np.inf, max_threshold:int=np.inf):
    """Clears the array with the generated factor after the get_factor function. Individual values are cleared when threshold values are exceeded.
     Threshold values are set according to the data of another factor. The shape of the factor matrices must be the same.

    Input:

        - factor (np.array): factor matrix after the get_factor function;

        - filter_factor(np.array): factor by which the thresholds are determined. Do not apply the function get_factor;

        - min_threshold (int): Values less than this threshold in the filter_factor array will be reset to zero in the factor array;

        - max_threshold (int): Values greater than this threshold in the filter_factor array will be reset to zero in the factor array;

    Output:

        - stock_universe (np.array): matrix with processed and cleaned factors for the subsequent formation of positions.
    """
    stock_universe = np.copy(factor)
    filter_factor = np.roll(filter_factor, np.shape(filter_factor)[1])
    filter_factor[0] = np.nan

    filter_factor = np.where((filter_factor > min_threshold) & (filter_factor < max_threshold), 1, 0)
    stock_universe *= filter_factor 
    stock_universe[stock_universe == 0] = np.nan
    
    return stock_universe

def set_holding_period(portfolio_positions:np.array, holding_period:int=1, looking_period:int=1, static:bool=True):
    """Sets the position holding for n periods in the position formation functions.

    Input:

        - portfolio_positions (np.array): binary matrix with positions for each security in each period;

        - holding_period (int): number of periods to hold a position after formation. Calculated from the first period 
        of the portfolio formation holding;
        
        - looking_period (int): Use for dynamic factors. Influences the variable n in the formula: t0/(t-n) to calculate factor change. 
        Influences the start period of the test. For static factors, the effect will be equivalent to setting a lag;

        - static (bool): static=True should be used for data for which it is not necessary to calculate the change. 
        For example, for P/E multipliers, EV/EBITDA, number of messages on Twitter etc.
        static=False counts the ratio to the previous period based on the looking period. 
        For example, for Momentum, changes in trading volume, number of messages on Twitter etc.
        The start of testing will be 1 period later than for static factors.

    Output:

        - rebalancing_periods (np.array): matrix with updated positions in the portfolio.
    """   
    n=0
    
    if static == True:
        n=1
        
    rebalancing_periods = np.empty(portfolio_positions.shape)
    rebalancing_periods[::] = np.nan
    rebalancing_periods[looking_period+n::holding_period] = portfolio_positions[looking_period+n::holding_period]
    
    mask = np.isnan(rebalancing_periods)
    idx = np.where(~mask,np.arange(mask.shape[0])[:,None],0)
    rebalancing_periods = np.take_along_axis(rebalancing_periods, np.maximum.accumulate(idx,axis=0),axis=0)

    return rebalancing_periods