"""
This module contains instruments to test, whether performance of a portfolio is statistically
significant or not. The process of statistical testing is an important part, when developing
a portfolio, because it is very easy to fall into the trap: you can get a portfolio with very high
total returns, but these returns are gotten randomly, so it is not likely that you wanna use this
strategy in future.
"""

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

from .portfolios import Portfolio, generate_random_portfolios

__all__ = [
    'zero_intelligence_test',
    'prophet_test',
    't_test',
]


def zero_intelligence_test(stock_prices, portfolio, target_metric, quantiles, **kwargs):
    """Creates random portfolios, replicating positions of `portfolio`.

    After creating random portfolios, calculates target for each of them. Then,
    selects `n_quantiles` portfolios by quantiles of target.

    Parameters
    ----------
    stock_prices : pd.DataFrame
        Prices, representing stock universe.
    portfolio : Portfolio
        Portfolio, to be tested (replicated by random portfolios).
    target_metric : callable
        Function-like object, computing some metric (e.g. value of metric). It must get as input 
        portfolio returns and return as output a number - value of the metric.
    quantiles : int > 1
        How many quantile-bounds to generate.
    **kwargs
        Keyword arguments for building up portfolios. See
        :func:`~pqr.portfolios.generate_random_portfolios`.

    Returns
    -------
    dict
        Dict, where keys are values of tops of quantiles of target metric and values are portfolios, achieving 
        these values.
    """

    random_portfolios = generate_random_portfolios(stock_prices, portfolio, **kwargs)
    target_values = pd.Series([target_metric(p.returns) for p in random_portfolios])

    indices = [target_values[target_values <= target_values.quantile(q)].argmax()
               for q in np.linspace(0, 1, quantiles)]
    target_quantiles = target_values[indices].sort_values()
    target_portfolios = np.array(random_portfolios)[target_quantiles.index]

    return {quantile: portfolio for quantile, portfolio in zip(target_quantiles, target_portfolios)}


def prophet_test(stock_prices, portfolio, mask=None):
    """Compare picks of a `portfolio` with "ideal" portfolio.

    "Ideal" portfolio is a portfolio, which always long stocks with the highest returns in the next period and
    short stocks with the lowest. It has the same amount of stocks in long and short as the given `portfolio`.
    The result of the test is represented as the mean share of matching picks to total picks.

    Parameters
    ----------
    stock_prices : pd.DataFrame
        Prices, representing stock universe.
    portfolio : Portfolio
        Portfolio to be tested (replicated by "ideal" portfolio).
    mask : pd.DataFrame, optional
        Matrix of True/False, where True means that a stock can be picked in that period and
        False - that a stock cannot be picked.

    Returns
    -------
    float
        Mean share of matching with "ideal" portfolio picks to total picks.
    """

    ideal_portfolio = Portfolio('ideal')
    ideal_portfolio.pick_ideally(stock_prices, portfolio, mask)

    match_in_picks = (ideal_portfolio.picks.replace(0, np.nan) == portfolio.picks.replace(0, np.nan)).sum(axis=1)
    total_picks = portfolio.picks[portfolio.picks != 0].sum(axis=1)

    return (match_in_picks / total_picks).mean()


def t_test(portfolio, risk_free_rate=0):
    """Calculates t-statistic and p-value of `portfolio` returns.

    Implements one-sided t-test. Null hypothesis is that `portfolio` returns are greater than
    `risk_free_rate`.

    Parameters
    ----------
    portfolio : Portfolio
        An allocated portfolio.
    risk_free_rate : array_like, default=0
        Indicative rate of guaranteed returns (e.g. US government bond rate).

    Returns
    -------
    tuple of float
        Tuple with results of t-test.
    """

    return ttest_1samp(portfolio.returns, risk_free_rate, alternative='greater')
