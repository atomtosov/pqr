"""
This module contains all necessary functionality to create factor models. Factor model is the set of
portfolios, covering all stock universe in each period. It is used to test whether any factor is
present on the market or not, that is why full coverage of stock universe is important.

Usually factor models are built for single-factors with the quantiles method. So, only this way to
fit factor models is natively supported, but if you need to make full coverage of stock universe
with time-series method or top-n portfolios, you can make it by yourself, using portfolios module.

Users of factor models are often interested in performance of the wml-portfolio, and this
functionality is also supported.

Moreover, you can calibrate factor model parameters, using simple bruteforce method, provided by
grid_search() function.
"""

import numpy as np
import pandas as pd

import pqr.portfolios

__all__ = [
    'fit_factor_model',
    'calculate_portfolios_summary_stats',
    'factor_model_tear_sheet',
    'grid_search',
]


def fit_factor_model(stock_prices, factor, weighting_factor=None, balance=None, fee_rate=0,
                     quantiles=3, add_wml=False, is_bigger_better=True):
    """
    Fits factor model with quantile-method.

    Creates `quantiles` portfolios, covering all stock universe and
    (optionally) add wml-portfolio.

    Parameters
    ----------
    stock_prices : pd.DataFrame
        Prices, representing stock universe.
    factor : Factor
        Factor to pick stocks into the portfolio.
    weighting_factor : Factor, optional
        Factor to weigh picks.
    balance : int or float, optional
        Initial balance of the portfolio.
    fee_rate : int or float, default=0
        Indicative commission rate for every deal.
    quantiles : int > 1, default=3
        Number of portfolios to build for covering stock universe by quantiles.
    add_wml : bool, default=False
        Whether to also create wml-portfolio or not.
    is_bigger_better : bool, default=True
        Whether portfolio with highest quantiles will be "winners" for
        wml-portfolio or with lowest ones.

    Returns
    -------
    list of Portfolio
        Factor portfolios, covering all stock universe (optionally, with wml-portfolio).
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
        wml = pqr.portfolios.Portfolio('wml')
        if is_bigger_better:
            wml.pick_stocks_wml(portfolios[-1], portfolios[0])
        else:
            wml.pick_stocks_wml(portfolios[0], portfolios[-1])
        if weighting_factor is not None:
            wml.weigh_by_factor(weighting_factor)
        else:
            wml.weigh_equally()
        wml.allocate(stock_prices, balance, fee_rate)

        portfolios.append(wml)

    return portfolios


def calculate_portfolios_summary_stats(portfolios, benchmark):
    """
    Calculates portfolios summary statistics and gather them into a table.

    See :func:`~pqr.metrics.summary`.

    Parameters
    ----------
    portfolios : Portfolio
        Portfolios, for which summary stats are calculated.
    benchmark : Portfolio or Benchmark
        Benchmark to compute some metrics.

    Returns
    -------
    pd.DataFrame
        Table with metrics for all given `portfolios`.
    """

    return pd.DataFrame([pqr.metrics.summary(p, benchmark) for p in portfolios]).T.round(2)


def factor_model_tear_sheet(portfolios, benchmark):
    """
    Shows the performance assessment of a factor model' portfolios.

    For now:

    * shows summary stats table
    * plots cumulative returns

    Parameters
    ----------
    portfolios : sequence of Portfolio
        Portfolios, included into the factor model.
    benchmark : Portfolio or Benchmark
        Benchmark to compute some metrics.

    Returns
    -------
    pd.DataFrame
        Table with summary stats.
    """

    stats = calculate_portfolios_summary_stats(portfolios, benchmark)
    pqr.plotting.plot_cumulative_returns(portfolios, benchmark)
    return stats


def grid_search(stock_prices, factor_data, looking_back_periods, method, lag_periods,
                holding_periods, benchmark, mask=None, **kwargs):
    """
    Fits a grid of factor models.

    Can be used to find the best parameters or just as fast alias to build a
    lot of models with different parameters.

    Parameters
    ----------
    stock_prices : pd.DataFrame
        Prices, representing stock universe.
    factor_data : pd.DataFrame
        Factor, used to pick stocks from (filtered) stock universe.
    looking_back_periods : sequence of int > 0
        Looking back periods for `factor` to be tested.
    method : {'static', 'dynamic'}
        Whether absolute values of `factor` are used to make decision or
        their percentage changes.
    lag_periods : sequence of int >= 0
        Lag periods for `factor` to be tested.
    holding_periods : sequence of int > 0
        Holding periods for `factor` to be tested.
    benchmark : Benchmark
        Benchmark to compare with portfolios and calculate some metrics.
    mask : pd.DataFrame, optional
        Matrix of True/False, where True means that a value should remain
        in `factor` and False - that a value should be deleted.
    **kwargs
        Keyword arguments for fitting factor models. See
        :func:`~pqr.factor_models.fit_factor_model`.

    Returns
    -------
    dict
        Dict, where key is the combination of looking back, lag and holding periods and value is a
        table with summary stats.
    """

    results = {}
    for looking in looking_back_periods:
        for lag in lag_periods:
            for holding in holding_periods:
                factor = pqr.factors.Factor(factor_data)
                factor.look_back(looking, method).lag(lag).hold(holding)
                if mask is not None:
                    factor.prefilter(mask)
                portfolios = fit_factor_model(stock_prices, factor, **kwargs)
                results[(looking, lag, holding)] = calculate_portfolios_summary_stats(portfolios,
                                                                                      benchmark)
    return results


def _split_quantiles(n):
    quantile_pairs = np.take(np.linspace(0, 1, n + 1),
                             np.arange(n * 2).reshape((n, -1)) -
                             np.indices((n, 2))[0])
    return [tuple(pair) for pair in quantile_pairs]
