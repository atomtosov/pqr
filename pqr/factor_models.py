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

from __future__ import annotations

from typing import Tuple, List, Optional, Iterable, Dict, Literal

import numpy as np
import pandas as pd

import pqr.benchmarks
import pqr.factors
import pqr.portfolios

__all__ = [
    'fit_factor_model',
    'calculate_portfolios_summary_stats',
    'factor_model_tear_sheet',
    'grid_search',
]


def fit_factor_model(
        stock_prices: pd.DataFrame,
        factor: pqr.factors.Factor,
        weighting_factor: Optional[pqr.factors.Factor] = None,
        balance: Optional[int | float] = None,
        fee_rate: int | float = 0,
        quantiles: int = 3,
        add_wml: bool = False,
        is_bigger_better: bool = True
) -> List[pqr.portfolios.AbstractPortfolio]:
    """
    Fits factor model with quantile-method.

    Creates `quantiles` portfolios, covering all stock universe and
    (optionally) add wml-portfolio.

    Parameters
    ----------
    stock_prices
        Prices, representing stock universe.
    factor
        Factor to pick stocks into the portfolio.
    weighting_factor
        Factor to weigh picks.
    balance
        Initial balance of the portfolio.
    fee_rate
        Indicative commission rate for every deal.
    quantiles
        Number of portfolios to build for covering stock universe by quantiles.
    add_wml
        Whether to also create wml-portfolio or not.
    is_bigger_better
        Whether portfolio with highest quantiles will be "winners" for
        wml-portfolio or with lowest ones.
    """

    quantiles_ = _split_quantiles(quantiles)

    portfolios = []
    for q in quantiles_:
        portfolio = pqr.portfolios.Portfolio('q({:.2f}, {:.2f})'.format(*q))
        portfolio.pick_stocks_by_factor(factor, q)

        if weighting_factor is not None:
            portfolio.weigh_by_factor(factor)
        else:
            portfolio.weigh_equally()

        portfolio.allocate(stock_prices, balance, fee_rate)

        portfolios.append(portfolio)

    if add_wml:
        if is_bigger_better:
            wml = pqr.portfolios.WmlPortfolio(portfolios[-1],
                                              portfolios[0])
        else:
            wml = pqr.portfolios.WmlPortfolio(portfolios[0],
                                              portfolios[-1])
        portfolios.append(wml)

    return portfolios


def calculate_portfolios_summary_stats(
        *portfolios: pqr.portfolios.AbstractPortfolio,
        benchmark: pqr.benchmarks.Benchmark
) -> pd.DataFrame:
    """
    Calculates portfolios summary statistics and gather them into a table.

    See function pqr.metrics.summary().

    Parameters
    ----------
    portfolios
        Portfolios, for which summary stats are calculated.
    benchmark
        Benchmark to compute some metrics.
    """

    stats = pd.DataFrame(
        [pqr.metrics.summary(p, benchmark) for p in portfolios]).T.round(2)
    return stats


def factor_model_tear_sheet(
        *portfolios: pqr.portfolios.AbstractPortfolio,
        benchmark: pqr.benchmarks.Benchmark,
) -> pd.DataFrame:
    """
    Shows the performance assessment of a factor model' portfolios.

    For now:

    * shows summary stats table
    * plots cumulative returns

    Parameters
    ----------
    portfolios
        Portfolios, included into the factor model.
    benchmark
        Benchmark to compute some metrics.
    """

    stats = calculate_portfolios_summary_stats(*portfolios,
                                               benchmark=benchmark)
    pqr.visualization.plot_cumulative_returns(*portfolios, benchmark=benchmark)
    return stats


def grid_search(
        stock_prices: pd.DataFrame,
        factor: pd.DataFrame,
        looking_back_periods: Iterable[int],
        method: Literal['static', 'dynamic'],
        lag_periods: Iterable[int],
        holding_periods: Iterable[int],
        benchmark: pqr.benchmarks.Benchmark,
        mask: Optional[pd.DataFrame] = None,
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
    looking_back_periods
        Looking back periods for `factor` to be tested.
    method
        Whether absolute values of `factor` are used to make decision or
        their percentage changes.
    lag_periods
        Lag periods for `factor` to be tested.
    holding_periods
        Holding periods for `factor` to be tested.
    benchmark
        Benchmark to compare with portfolios and calculate some metrics.
    mask
        Matrix of True/False, where True means that a value should remain
        in `factor` and False - that a value should be deleted.
    **kwargs
        Keyword arguments for fitting factor models. See fit_factor_model().
    """

    results = {}
    for looking in looking_back_periods:
        for lag in lag_periods:
            for holding in holding_periods:
                transformed_factor = pqr.factors.Factor(factor)
                transformed_factor.transform(looking, method, lag, holding)
                if mask is not None:
                    transformed_factor.prefilter(mask)
                portfolios = fit_factor_model(stock_prices, transformed_factor,
                                              **kwargs)
                results[(looking, lag, holding)] = (
                    calculate_portfolios_summary_stats(*portfolios,
                                                       benchmark=benchmark)
                )
    return results


def _split_quantiles(n: int) -> List[Tuple[int | float, int | float]]:
    """
    Creates `n` quantiles, covering all range from 0 to 1.
    """

    quantile_pairs = np.take(np.linspace(0, 1, n + 1),
                             np.arange(n * 2).reshape((n, -1)) -
                             np.indices((n, 2))[0])
    return [tuple(pair) for pair in quantile_pairs]
