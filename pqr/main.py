import numpy as np
from numpy.lib.function_base import quantile
import pandas as pd
from datetime import datetime
np.set_printoptions(suppress=True)

looking_period = 1
lag = 1
holding_period = 1
quantile_step = 0.2
liquidity_threshold = 100_000_000 
fee_rate= 0.0005

stock_list = ['AAPL', 'FB', 'NFLX', 'GOOG', 'AMZN', 'DAL', 'GE', 'XOM', 'F', 'WFC']
start_date = datetime(2016,1,1) # YYYY-MM-DD
end_date = datetime(2021,4,30)

import pandas_datareader.data as web

stocks_request = web.DataReader(stock_list, 'yahoo', start_date, end_date)
stocks_price = pd.DataFrame(stocks_request['Adj Close'])
monthly_price = stocks_price.resample(rule='M').last()

stocks_volume = pd.DataFrame(stocks_request['Volume'])
stocks_volume *= stocks_price
monthly_volume = stocks_volume.resample(rule='M').mean()

price = monthly_price.values
volume = monthly_volume.values
dates = np.array(monthly_price.index)

from pqr.data_preprocessing import get_factor
momentum_factor = get_factor(price, static=False, looking_period=looking_period, lag=lag)

from pqr.data_preprocessing import set_stock_universe
momentum_factor = set_stock_universe(momentum_factor, volume, min_threshold=liquidity_threshold)

from pqr.portfolios_formation import get_all_quantiles
momentum_positions, momentum_names = get_all_quantiles(momentum_factor, quantile_step=quantile_step, holding_period=holding_period, static=False)

from pqr.portfolios_formation import set_equal_weights
momentum_portfolio = set_equal_weights(momentum_positions)

from pqr.costs import get_fee_costs
fee = get_fee_costs(momentum_portfolio, fee_rate=fee_rate)

from pqr.portfolio_return import get_universe_return
from pqr.portfolio_return import get_portfolio_return
universe_return = get_universe_return(price)
momentum_portfolio_return = get_portfolio_return(momentum_portfolio, universe_return, fee_lists=fee)

from pqr.portfolio_return import compare_portfolios
from pqr.benchmark import get_benchmark
benchmark_return = get_benchmark(price, momentum_factor, universe_return, equal_weights=True)

momentum_results = compare_portfolios(momentum_names, momentum_portfolio_return, benchmark_return, dates)