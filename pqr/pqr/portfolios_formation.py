import numpy as np
from pqr.data_preprocessing import set_holding_period

def get_ts_positions(factor:np.array, holding_period:int=1, threshold:int=0, static:bool=True):
    """Forms positions on the selected factor into two portfolios - winners and losers. The decision is made based on 
    the dynamics of the factor to its own past values and exceeding the threshold. In contrast to the cross section, 
    the number of shares in the portfolios can be highly unbalanced.

    Input:

        - factor (np.array): np.array with processed factor after function get_factor;

        - holding_period (int): number of periods to hold a position after formation. Calculated from the first period 
        of the portfolio formation holding;

        - threshold (int): value that must be exceeded in order for the instrument to get into the winners' portfolio. 
        Otherwise, it enters the portfolio of losers;

        - static (bool): static=True should be used for data for which it is not necessary to calculate the change. 
        For example, for P/E multipliers, EV/EBITDA, number of messages on Twitter etc.
        static=False counts the ratio to the previous period based on the looking period. 
        For example, for Momentum, changes in trading volume, number of messages on Twitter etc.
        The start of testing will be 1 period later than for static factors.

    Output:

        - positions_arrays (list): list with binary arrays for each instrument for each period in the portfolio;
        
        - quantile_names (list): list with the names of the portfolios. The composition and names of the lists are constant.
    """     
    quantile_names = ['ts_winners', 'ts_losers']
    positions_arrays = []
    
    ts_winners = np.where((factor > threshold), 1, 0)
    ts_losers = np.where((factor < threshold), 1, 0)
    
    if holding_period != 1:
        
        ts_winners = set_holding_period(ts_winners, holding_period=holding_period, static=static)
        ts_losers = set_holding_period(ts_losers, holding_period=holding_period, static=static)
        
    positions_arrays.append(ts_winners)
    positions_arrays.append(ts_losers)
    
    return positions_arrays, quantile_names

def get_all_quantiles(factor:np.array, quantile_step:int=0.25, holding_period:int=1, static:bool=True):
    """Forms positions on the selected factor into equal n portfolios. n is defined as 100% / quantile_step. 
    The 1st quantile [0%:quantile_step] represents the portfolio with the lowest quantile_step % values of the factors for each period. 
    The portfolio [quantile_step:100%] includes the stocks with the highest factor value

    Input:

        - factor (np.array): np.array with processed factor after function get_factor;

        - quantile_step (int): step, which determines the number of portfolios and the number of instruments in each of them;

        - holding_period (int): number of periods to hold a position after formation. Calculated from the first period 
        of the portfolio formation holding;

        - static (bool): static=True should be used for data for which it is not necessary to calculate the change. 
        For example, for P/E multipliers, EV/EBITDA, number of messages on Twitter etc.
        static=False counts the ratio to the previous period based on the looking period. 
        For example, for Momentum, changes in trading volume, number of messages on Twitter etc.
        The start of testing will be 1 period later than for static factors.

    Output:

        - positions_arrays (list): list with binary arrays for each instrument for each period in the portfolio;
        
        - quantile_names (list): list with the names of the portfolios. The composition and names of the lists are constant.
    """     
    quantile_list = []
    quantile_names = []
    positions_names = [] 
    positions_arrays = []
    

    for i in range(int(100/(quantile_step*100))):
        
        percentile = round((i+1)*quantile_step,2)
        quantile_list.append(percentile)
        quantile_names.append(int(round(percentile*100,0)))

    for i in range(len(quantile_list)):
    
        if i ==0:
            
            threshold = np.nanquantile(factor, quantile_list[i], axis=1)
            positions_arrays.append(np.where((factor < threshold[:,None]), 1, 0))
            positions_names.append('quantile_%s' % (quantile_names[i]))
            previous_threshold = threshold
            continue
            
        if quantile_list[i] == quantile_list[-1]:
            
            positions_arrays.append(np.where((factor > previous_threshold[:,None]), 1, 0))
            positions_names.append('quantile_%s' % (quantile_names[i]))
                      
            if holding_period != 1:
                for i in range(len(positions_names)):
                    positions_arrays[i] = set_holding_period(positions_arrays[i], holding_period=holding_period, static=static)
            break
            
        threshold = np.nanquantile(factor, quantile_list[i], axis=1)
        positions_arrays.append(np.where((factor > previous_threshold[:,None]) & (factor < threshold[:,None]), 1, 0))
        positions_names.append('quantile_%s' % (quantile_names[i]))
        previous_threshold = threshold
        
    
    return  positions_arrays, positions_names

def set_equal_weights(positions_lists:list, benchmark:bool=False):
    """Determines the equal weights of positions in the portfolio. 
    The weight depends on the number of positions in each period and can vary from period to period

    Input:

        - positions_lists (list): list of binary matrices with positions of instruments in the portfolio;

        - benchmark (bool): benchmark==True is set to feed a single array, not a list. Specifically, the benchmark.

    Output:
        
        - portfolio_weights (list): list with portfolio weights. The sum of weights for each period always equals 1.
    """   
    portfolio_weights = []
    
    for i in range(len(positions_lists)):
        
        if benchmark==True:
            positions_sum = positions_lists.sum(axis=1)
            portfolio_weights = positions_lists / positions_sum[:,None]
            break
            
        positions_sum = positions_lists[i].sum(axis=1)
        portfolio_weights.append(positions_lists[i] / positions_sum[:,None])

    return portfolio_weights

def set_value_weights(positions_lists:list, weight_factor:np.array, benchmark:bool=False):
    """Determines the factor weights of positions in the portfolio. 
    The weight depends on the number of positions and weight_factor values.
    Weights consider only positive values. Negative variables will be equated to 0

    Input:

        - positions_lists (list): list of binary matrices with positions of instruments in the portfolio;

        - weight_factor (np.array): Matrix with the numerical values of the selected weight factor. For example, Market Capitalization;

        - benchmark (bool): benchmark==True is set to feed a single array, not a list. Specifically, the benchmark.

    Output:
        
        - portfolio_weights (list): list with portfolio weights. The sum of weights for each period always equals 1.
    """      
    portfolio_weights = []
    
    nans = np.isnan(weight_factor)
    weight_factor[nans] = 0
    weight_factor = np.roll(weight_factor, np.shape(weight_factor)[1])
    weight_factor[0] = np.nan    
    
    for i in range(len(positions_lists)):
        
        if benchmark==True:
            portfolio_weight_factors = weight_factor * positions_lists
            portfolio_weight_factors_sum = np.nansum(portfolio_weight_factors, axis=1)
            portfolio_weights = portfolio_weight_factors / portfolio_weight_factors_sum[:,None]
            break   
 
        portfolio_weight_factors = weight_factor * positions_lists[i]
        portfolio_weight_factors_sum = np.nansum(portfolio_weight_factors, axis=1)
        portfolio_weights.append(portfolio_weight_factors / portfolio_weight_factors_sum[:,None])

    return portfolio_weights