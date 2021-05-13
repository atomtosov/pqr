import numpy as np

def get_fee_costs(portfolios_lists:list, fee_rate:int=0):
    """Calculates commissions for each trade.

    Input:

        - positions_lists (list): list of matrices with weights of instruments in the portfolio;

        - fee_rate (int): indicative commission of the exchange and the broker for each trade.

    Output:
        
        - portfolio_fees (list): list with portfolio costs. The sum of weights for first period always equals 1.
        The sum of the remaining periods from 0 to 2.
    """        
    portfolio_fees = []
        
    for i in range(len(portfolios_lists)):  
        
        sum_row = portfolios_lists[i].sum(axis=1)
        first_row_index = np.nonzero(sum_row == 1)[0][0]
        first_row = portfolios_lists[i][first_row_index]

        fee = np.diff(portfolios_lists[i], axis=0)
        fee = np.insert(fee, first_row_index, first_row, axis=0)
            
        fee = np.absolute(fee)
        portfolio_fees.append(fee * fee_rate)
    
    return portfolio_fees
