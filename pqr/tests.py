"""
This module contains instruments to test, whether performance of a portfolio is
statistically significant or not. The process of statistical testing is
significant part, when developing a portfolio, because it is very easy to fall
into the trap: you can get a portfolio with very high total returns, but these
returns are gotten randomly, so it is not likely that you wanna use this
strategy in future.

For now, it has only 1 test - random test, but more will be added soon.
"""

from typing import Tuple, Callable, Union

import numpy as np
import pandas as pd

import pqr.portfolios

__all__ = [
    'zero_intelligence_test',
]


def zero_intelligence_test(stock_prices: pd.DataFrame,
                           portfolio: pqr.portfolios.Portfolio,
                           target_metric: Callable[
                               [pqr.portfolios.AbstractPortfolio],
                               Union[int, float]],
                           quantiles: int = 10,
                           **kwargs) -> Tuple[pd.Series, np.ndarray]:
    """
    Creates random portfolios, replicating positions of `portfolio`.

    After creating random portfolios, calculates target for each of them. Then,
    selects `n_quantiles` portfolios by quantiles of target.

    Parameters
    ----------
    stock_prices
        Prices, representing stock universe.
    portfolio
        Portfolio, to be tested (replicated by random portfolios).
    target_metric
        Function-like object, computing some number (e.g. value of metric). It
        must get Portfolio and return int or float.
    quantiles
        How many quantile-bounds to generate.
    **kwargs
        Keyword arguments for building up portfolios. See random_portfolios().
    """

    random_portfolios = pqr.portfolios.generate_random_portfolios(
        stock_prices,
        portfolio,
        **kwargs)

    target_values = pd.Series(
        [target_metric(p) for p in random_portfolios])

    indices = []
    for q in np.linspace(0, 1, quantiles):
        indices.append(
            target_values[target_values <= target_values.quantile(q)].argmax())

    target_quantiles = target_values[indices].sort_values()

    return (target_quantiles,
            np.array(random_portfolios)[target_quantiles.index])
