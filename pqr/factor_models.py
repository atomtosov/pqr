"""
This module contains all necessary functionality to create factor models.
Factor model is the set of portfolios, covering all stock universe in each
period. It is used to test whether any factor is present on the market or not,
that is why full coverage of stock universe is important.

Usually factor models are built for single-factors with the quantiles method.
So, only this way to fit factor models is natively supported, but if you need
to make full coverage of stock universe with time-series method or top-n
portfolios, you can make it by yourself, using portfolios module.

Users of factor models are often interested in performance of the
wml-portfolio, and this functionality is also supported.

Moreover, you can calibrate factor model parameters, using simple bruteforce
method, provided by grid_search() function.
"""

from typing import Iterable, List, Dict, Tuple

import numpy as np
import pandas as pd

import pqr.benchmarks
import pqr.factors
import pqr.metrics
import pqr.portfolios
import pqr.thresholds

__all__ = [
    'fit_factor_model',
    'compare_portfolios',
    'grid_search',
]


def fit_factor_model(
        stock_prices: pd.DataFrame,
        factor: pd.DataFrame,
        n_quantiles: int = 3,
        add_wml: bool = False,
        is_bigger_better: bool = True,
        **kwargs
) -> List[pqr.portfolios.Portfolio]:
    """
    Creates factor portfolios with quantiles method, covering stock universe.

    Parameters
    ----------
    stock_prices
        Prices, representing stock universe.
    factor
        Factor, used to pick stocks from (filtered) stock universe.
    n_quantiles
        Number of portfolios to build for covering stock universe by quantiles.
    add_wml
        Whether to also create wml-portfolio or not.
    is_bigger_better
        Whether portfolio with highest quantiles will be "winners" for
        wml-portfolio or with lowest ones.
    **kwargs
        Keyword arguments for building portfolios. See factor_portfolio().
    """

    portfolios = [
        pqr.portfolios.factor_portfolio(
            stock_prices,
            factor=factor, thresholds=q,
            name=f'q({q.lower:.2f}, {q.upper:.2f})',
            **kwargs
        )
        for q in _make_quantiles(n_quantiles)
    ]

    if add_wml:
        if is_bigger_better:
            wml = pqr.portfolios.wml_portfolio(portfolios[-1], portfolios[0])
        else:
            wml = pqr.portfolios.wml_portfolio(portfolios[0], portfolios[-1])
        portfolios.append(wml)

    return portfolios


def compare_portfolios(
        *portfolios: pqr.portfolios.Portfolio,
        benchmark: pqr.benchmarks.Benchmark
) -> pd.DataFrame:
    """
    Calculates summary statistics for portfolios.

    Parameters
    ----------
    portfolios
        Portfolios to be compared.
    benchmark
        Benchmark to calculate some metrics.
    """

    return pd.DataFrame(
        [pqr.metrics.summary(p, benchmark) for p in portfolios]
    ).T.round(2)


def grid_search(
        stock_prices: pd.DataFrame,
        factor: pd.DataFrame,
        is_dynamic: bool,
        looking_periods: Iterable[int],
        lag_periods: Iterable[int],
        holding_periods: Iterable[int],
        benchmark: pqr.benchmarks.Benchmark,
        **kwargs
) -> Dict[Tuple[int, int, int], pd.DataFrame]:
    """
    Fits a grid of factor models.

    Can be used to find the best parameters or just as fast alias to build a
    lot of models with different parameters.

    Parameters
    ----------
    stock_prices
        Prices, representing stock universe.
    factor
        Factor, used to pick stocks from (filtered) stock universe.
    is_dynamic
    looking_periods
        Looking periods for `factor` to be tested.
    lag_periods
        Lag periods for `factor` to be tested.
    holding_periods
        Holding periods for `factor` to be tested.
    benchmark
        Benchmark to compare with portfolios and calculate some metrics.
    **kwargs
        Keyword arguments for fitting factor models. See fit_factor_model().
    """

    results = {}
    for looking in looking_periods:
        for lag in lag_periods:
            for holding in holding_periods:
                transformed_factor = pqr.factors.factorize(factor, is_dynamic,
                                                           looking, lag,
                                                           holding)
                portfolios = fit_factor_model(stock_prices,
                                              transformed_factor,
                                              **kwargs)
                results[(looking, lag, holding)] = compare_portfolios(
                    *portfolios, benchmark=benchmark)
    return results


def _make_quantiles(n: int) -> List[pqr.thresholds.Quantiles]:
    """
    Creates `n` quantiles, covering all range from 0 to 1.
    """

    quantile_pairs = np.take(np.linspace(0, 1, n + 1),
                             np.arange(n * 2).reshape((n, -1)) -
                             np.indices((n, 2))[0])
    return [pqr.thresholds.Quantiles(*pair)
            for pair in quantile_pairs]
