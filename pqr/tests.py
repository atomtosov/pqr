from typing import Optional, Callable, Union

import pandas as pd
import numpy as np

import pqr.portfolios
import pqr.benchmarks
import pqr.factors
import pqr.metrics


def random_test(portfolio: pqr.portfolios.Portfolio,
                target: Callable[[pqr.portfolios.AbstractPortfolio],
                                 Union[int, float]],
                prices: pd.DataFrame,
                n_random_portfolios: int = 100,
                n_quantiles: int = 10,
                random_seed: Optional[int] = None,
                **kwargs):
    np.random.seed(random_seed)
    target_values = pd.Series(index=range(n_random_portfolios))
    random_portfolios = [None for _ in range(n_random_portfolios)]
    for i in range(n_random_portfolios):
        random_portfolio = pqr.portfolios.RandomPortfolio()
        random_portfolio.invest(prices, portfolio, target, **kwargs)
        target_values.values[i] = random_portfolio.target_value
        random_portfolios[i] = random_portfolio

    indices = []
    for q in np.linspace(0, 1, n_quantiles):
        indices.append(
            target_values[target_values <= target_values.quantile(q)].argmax())

    target_quantiles = target_values[indices].sort_values()

    return (target_quantiles,
            np.array(random_portfolios)[target_quantiles.index])


def _generate_random_factor(portfolio):
    random_data = pd.DataFrame(np.random.random(portfolio.positions.shape),
                               index=portfolio.positions.index,
                               columns=portfolio.positions.columns)
    return pqr.factors.Factor(random_data)
