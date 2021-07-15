from typing import Optional, Callable, Union

import pandas as pd
import numpy as np

import pqr.portfolios
import pqr.benchmarks
import pqr.factors
import pqr.metrics


def random_test(portfolio: pqr.portfolios.Portfolio,
                target: Callable[[pqr.portfolios.Portfolio],
                                 Union[int, float]],
                prices: pd.DataFrame,
                n_random_portfolios: int = 100,
                random_seed: Optional[int] = None,
                **kwargs):
    np.random.seed(random_seed)
    target_values = pd.Series(index=range(n_random_portfolios))
    for i in range(n_random_portfolios):
        random_factor = _generate_random_factor(portfolio)
        random_portfolio = pqr.portfolios.Portfolio(portfolio.balance,
                                                    portfolio.fee_rate)
        random_portfolio.invest(prices=prices,
                                picking_factor=random_factor,
                                picking_thresholds=portfolio.thresholds,
                                **kwargs)
        target_values.values[i] = target(random_portfolio)

    return target_values


def _generate_random_factor(portfolio):
    random_data = np.random.random(portfolio.positions.shape)
    random_data = pd.DataFrame(random_data,
                               index=portfolio.positions.index,
                               columns=portfolio.positions.columns)
    return pqr.factors.Factor(random_data)
