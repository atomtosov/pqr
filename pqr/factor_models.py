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

from .portfolios import Portfolio
from .factors import Factor
from .metrics import summary
from .plotting import plot_cumulative_returns

__all__ = [
    'fit_factor_model',
    'factor_model_tear_sheet',
    'grid_search',
]


def fit_factor_model(stock_prices, factor, better='less', weighting_factor=None, 
                     balance=None, fee_rate=0, quantiles=3, add_wml=False):
    """Fits factor model with quantile-method.

    Creates `quantiles` portfolios, covering all stock universe and (optionally) add wml-portfolio.

    Parameters
    ----------
    stock_prices : pd.DataFrame
        Prices, representing stock universe.
    factor : Factor
        Factor to pick stocks into the portfolio.
    better: {'more', 'less'}, default='more'
        Whether bigger values of factor are treated as better to pick or in contrary as better to 
        avoid. 
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

    Returns
    -------
    list of Portfolio
        Factor portfolios, covering all stock universe (optionally, with wml-portfolio).
    """

    quantiles_ = _split_quantiles(quantiles)

    portfolios = []
    for q in quantiles_:
        portfolio = Portfolio('q({:.2f}, {:.2f})'.format(*q))
        portfolio.pick_by_factor(factor, q, better, method='quantile')
        if weighting_factor is not None:
            portfolio.weigh_by_factor(factor)
        else:
            portfolio.weigh_equally()
        portfolio.allocate(stock_prices, balance, fee_rate)

        portfolios.append(portfolio)

    if add_wml:
        wml = Portfolio('wml')
        wml.pick_wml(portfolios[0], portfolios[-1])
        if weighting_factor is not None:
            wml.weigh_by_factor(weighting_factor)
        else:
            wml.weigh_equally()
        wml.allocate(stock_prices, balance, fee_rate)

        portfolios.append(wml)

    return portfolios


def factor_model_tear_sheet(portfolios, benchmark):
    """Shows the performance assessment of a factor model' portfolios.

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

    stats = pd.DataFrame([summary(p, benchmark) for p in portfolios]).T.round(2)
    plot_cumulative_returns(portfolios, benchmark)
    return stats


def grid_search(stock_prices, factor_data, params, target_metric, method='static', mask=None, **kwargs):
    """Fits a grid of factor models.

    Can be used to find the best parameters or just as fast alias to build a
    lot of models with different parameters.

    Parameters
    ----------
    stock_prices : pd.DataFrame
        Prices, representing stock universe.
    factor_data : pd.DataFrame
        Factor, used to pick stocks from (filtered) stock universe.
    params : sequence of tuple of int
        Parameters to iterate over `factor_data` to make it a factor.
    target_metric : callable
        Function-like object, computing some metric (e.g. value of metric). It must get as input 
        portfolio returns and return as output a number - value of the metric.
    method : {'static', 'dynamic'}, default='static'
        Whether absolute values of `factor` are used to make decision or their percentage changes.
    mask : pd.DataFrame, optional
        Matrix of True/False, where True means that a value should remain in `factor` and False - 
        that a value should be deleted.
    **kwargs
        Keyword arguments for fitting factor models. See :func:`~pqr.factor_models.fit_factor_model`.

    Returns
    -------
    pd.DataFrame
        Table, where index shows different combinations of given `params`, columns - names of portfolios
        and on cells values of `target_metric` are presented.
    """

    metric_rows = []
    for looking, lag, holding in params:
        factor = Factor(factor_data)
        factor.look_back(looking, method).lag(lag).hold(holding)
        if mask is not None:
            factor.filter(mask)
        portfolios = fit_factor_model(stock_prices, factor, **kwargs)
        
        metric_values = pd.DataFrame(
            [[target_metric(portfolio.returns) for portfolio in portfolios]],
            index=[(looking, lag, holding)], columns=[portfolio.name for portfolio in portfolios]
        )

        metric_rows.append(metric_values)

    return pd.concat(metric_rows)


def _split_quantiles(n):
    quantile_pairs = np.take(np.linspace(0, 1, n + 1),
                             np.arange(n * 2).reshape((n, -1)) -
                             np.indices((n, 2))[0])
    return [tuple(pair) for pair in quantile_pairs]
