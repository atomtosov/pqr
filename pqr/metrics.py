"""
This module contains metrics and statistics to assess performance of a
portfolio. Usual metrics are always numbers (int or float), but also rolling
metrics are supported. Rolling metrics are calculated to estimate annual
performance of a portfolio in each trading period: every period gather not all
points of returns but only for 1 year (e.g. if returns are monthly, rolling
window size is 12).

For now practically all popular in portfolio management metrics and statistics
are supported, but you are welcome to create your own metrics and contribute to
source code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.regression.linear_model as sm_linear
import statsmodels.regression.rolling as sm_rolling
import statsmodels.tools.tools as sm_tools

__all__ = [
    'summary',
    'cumulative_returns',
    'alpha', 'rolling_alpha',
    'beta', 'rolling_beta',
    'sharpe', 'rolling_sharpe',
    'mean_return', 'rolling_mean_return',
    'excess_return', 'rolling_excess_return',
    'volatility', 'rolling_volatility',
    'benchmark_correlation', 'rolling_benchmark_correlation',
    'win_rate', 'rolling_win_rate',
    'max_drawdown', 'rolling_max_drawdown',
]


def summary(
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
) -> pd.Series:
    """
    Calculates summary statistics for a `portfolio`.

    Computed metrics:

    - Alpha, %
    - Beta, %
    - Sharpe Ratio
    - Mean Return, %
    - Excess Return, %
    - Volatility, %
    - Benchmark Correlation, %
    - Win Rate, %
    - Maximum Drawdown, %

    Parameters
    ----------
    portfolio_returns
        Portfolio, for which metrics are calculated.
    benchmark_returns
        Benchmark, which used as the alternative for the `portfolio` to
        calculate some metrics (usually some stock index).
    """

    return pd.Series(
        {
            'Alpha, %': alpha(portfolio_returns, benchmark_returns) * 100,
            'Beta': beta(portfolio_returns, benchmark_returns),
            'Sharpe Ratio': sharpe(portfolio_returns),
            'Mean Return, %': mean_return(portfolio_returns) * 100,
            'Mean Excessive Return, %': excess_return(portfolio_returns,
                                                      benchmark_returns) * 100,
            'Volatility, %': volatility(portfolio_returns) * 100,
            'Benchmark Correlation': benchmark_correlation(portfolio_returns,
                                                           benchmark_returns),
            'Win Rate, %': win_rate(portfolio_returns) * 100,
            'Maximum Drawdown, %': max_drawdown(portfolio_returns) * 100,
        },
        name=portfolio_returns.name
    ).round(2)


def cumulative_returns(returns: pd.Series) -> pd.Series:
    return (returns + 1).cumprod() - 1


def alpha(
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
) -> int | float:
    """
    Calculates alpha of a `portfolio`.

    Alpha is the coefficient :math:`\\alpha` in the estimated regression per
    benchmark returns:

    .. math:: r_{portfolio} = \\alpha + \\beta * r_{benchmark} + \\epsilon

    Parameters
    ----------
    portfolio_returns
        Portfolio, for which alpha is calculated.
    benchmark_returns
        Benchmark, which used as the alternative for the `portfolio`.
    """

    est = _estimate_ols(portfolio_returns, benchmark_returns)
    return est.params[0]


def rolling_alpha(
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
) -> pd.Series:
    """
    Calculates rolling on 1 trading year alpha of a `portfolio`.

    See function alpha.

    Parameters
    ----------
    portfolio_returns
        Portfolio, for which rolling alpha is calculated.
    benchmark_returns
        Benchmark, which used as the alternative for the `portfolio`.
    """

    est = _estimate_rolling_ols(portfolio_returns, benchmark_returns)
    return pd.Series(est.params[:, 0], index=portfolio_returns.index)


def beta(
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
) -> int | float:
    """
    Calculates beta of a `portfolio`.

    Beta is the coefficient :math:`\\beta` in the estimated regression per
    benchmark returns:

    .. math:: r_{portfolio} = \\alpha + \\beta * r_{benchmark} + \\epsilon


    Parameters
    ----------
    portfolio_returns
        Portfolio, for which beta is calculated.
    benchmark_returns
        Benchmark, which used as the alternative for the `portfolio`.
    """

    est = _estimate_ols(portfolio_returns, benchmark_returns)
    return est.params[1]


def rolling_beta(
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
) -> pd.Series:
    """
    Calculates rolling on 1 trading year beta of a `portfolio`.

    See function beta.

    Parameters
    ----------
    portfolio_returns
        Portfolio, for which rolling beta is calculated.
    benchmark_returns
        Benchmark, which used as the alternative for the `portfolio`.
    """

    est = _estimate_rolling_ols(portfolio_returns, benchmark_returns)
    return pd.Series(est.params[:, 1], index=portfolio_returns.index)


def sharpe(
        portfolio_returns: pd.Series,
        risk_free_rate: int | float = 0
) -> int | float:
    """
    Calculates sharpe ratio of a `portfolio`.

    Sharpe Ratio is the ratio of mean return and standard deviation of
    adjusted returns:

    .. math:: SR = \\frac{\\bar{r}}{\\sigma_{r}}

    Parameters
    ----------
    portfolio_returns
        Portfolio, for which sharpe ratio is calculated.
    risk_free_rate
    """

    annualization_rate = np.sqrt(_get_freq(portfolio_returns))
    portfolio_returns = portfolio_returns - risk_free_rate
    mean = mean_return(portfolio_returns)
    std = volatility(portfolio_returns)
    return mean / std * annualization_rate


def rolling_sharpe(
        portfolio_returns: pd.Series,
        risk_free_rate: int | float = 0
) -> pd.Series:
    """
    Calculates rolling on 1 trading year sharpe ratio of a `portfolio`.

    See function sharpe.

    Parameters
    ----------
    portfolio_returns
        Portfolio, for which rolling sharpe ratio is calculated.
    risk_free_rate
    """

    annualization_rate = np.sqrt(_get_freq(portfolio_returns))
    portfolio_returns = portfolio_returns - risk_free_rate
    mean = rolling_mean_return(portfolio_returns)
    std = rolling_volatility(portfolio_returns)
    return mean / std * annualization_rate


def mean_return(portfolio_returns: pd.Series) -> int | float:
    """
    Calculates mean return of a `portfolio`.

    Mean Return is simple expected value:

    .. math:: \\bar{r} = \\frac{\\sum_{i=1}^{n}r}{n}

    Parameters
    ----------
    portfolio_returns
        Portfolio, for which mean return is calculated.
    """

    return portfolio_returns.mean()


def rolling_mean_return(portfolio_returns: pd.Series) -> pd.Series:
    """
    Calculates rolling on 1 trading year mean return of a `portfolio`.

    See function mean_return.

    Parameters
    ----------
    portfolio_returns
        Portfolio, for which mean return is calculated.
    """

    freq = _get_freq(portfolio_returns)
    return portfolio_returns.rolling(freq).mean()


def excess_return(
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
) -> int | float:
    """
    Calculates mean excessive return of a `portfolio`.

    Mean Excessive Return is difference between mean return of portfolio and
    mean return of benchmark:

    .. math:: MER = \\overline{r_{portfolio}} - \\overline{r_{benchmark}}

    Parameters
    ----------
    portfolio_returns
        Portfolio, for which alpha is calculated.
    benchmark_returns
        Benchmark, which used as the alternative for the `portfolio`.
    """

    mean_portfolio_return = mean_return(portfolio_returns)
    mean_benchmark_return = mean_return(benchmark_returns)
    return mean_portfolio_return - mean_benchmark_return


def rolling_excess_return(
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
) -> pd.Series:
    """
    Calculates rolling on 1 trading year mean excessive return of portfolio.

    See function mean_excessive_return.

    Parameters
    ----------
    portfolio_returns
        Portfolio, for which rolling mean excessive return is calculated.
    benchmark_returns
        Benchmark, which used as the alternative for the `portfolio`.
    """

    mean_portfolio_return = rolling_mean_return(portfolio_returns)
    mean_benchmark_return = rolling_mean_return(benchmark_returns)
    return mean_portfolio_return - mean_benchmark_return


def volatility(portfolio_returns: pd.Series) -> int | float:
    """
    Calculates volatility of returns of a `portfolio`.

    Volatility of the portfolio is standard deviation of portfolio returns:

    .. math:: \\sigma_r=\\sqrt{\\frac{\\sum_{i=1}^{n}(r_i-\\bar{r})^2}{n-1}}

    Parameters
    ----------
    portfolio_returns
        Portfolio, for which volatility is calculated.
    """

    return portfolio_returns.std()


def rolling_volatility(portfolio_returns: pd.Series) -> pd.Series:
    """
    Calculates rolling on 1 trading year volatility of a `portfolio`.

    See function volatility.

    Parameters
    ----------
    portfolio_returns
        Portfolio, for which volatility is calculated.
    """

    freq = _get_freq(portfolio_returns)
    return portfolio_returns.rolling(freq).std()


def benchmark_correlation(
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
) -> int | float:
    """
    Calculates correlation between a `portfolio` and a `benchmark`.

    Benchmark Correlation is simple correlation between 2 time-series - returns
    of portfolio and returns of benchmark:

    .. math:: BC = corr(r_{portfolio}, r_{benchmark})

    Parameters
    ----------
    portfolio_returns
        Portfolio, for which correlation with the `benchmark` is calculated.
    benchmark_returns
        Benchmark, which used as the alternative for the `portfolio`.
    """

    return portfolio_returns.corr(benchmark_returns)


def rolling_benchmark_correlation(
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
) -> pd.Series:
    """
    Calculates rolling on 1 trading year correlation between portfolio and
    benchmark.

    See function benchmark_correlation.

    Parameters
    ----------
    portfolio_returns
        Portfolio, for which rolling correlation with the `benchmark` is
        calculated.
    benchmark_returns
        Benchmark, which used as the alternative for the `portfolio`.
    """

    freq = _get_freq(portfolio_returns.returns)
    return portfolio_returns.returns.rolling(freq).corr(
        benchmark_returns.returns)


def win_rate(portfolio_returns: pd.Series) -> int | float:
    """
    Calculates share of profitable periods of a `portfolio`.

    Share of Profitable Periods of portfolios is simple ratio of number of
    periods with positive returns and total number of trading periods:

    .. math:: SPR = \\frac{\\sum_{i=1}^{n}[r_{i} > 0]}{n}

    Parameters
    ----------
    portfolio_returns
        Portfolio, for which share of profitable periods is calculated.
    """

    positive_periods = (portfolio_returns > 0).sum()
    total_periods = np.size(portfolio_returns)
    return positive_periods / total_periods


def rolling_win_rate(portfolio_returns: pd.Series) -> pd.Series:
    """
    Calculates rolling on 1 trading year share of profitable periods of a
    `portfolio`.

    See function profitable_periods_share.

    Parameters
    ----------
    portfolio_returns
        Portfolio, for which share of profitable periods is calculated.
    """

    freq = _get_freq(portfolio_returns.returns)
    rolling_positive = (portfolio_returns.returns > 0).rolling(freq).sum()
    return rolling_positive / freq


def max_drawdown(portfolio_returns: pd.Series) -> int | float:
    """
    Calculates maximum drawdown of a `portfolio`.

    Maximum Drawdown of portfolio is the highest relative difference between
    local maximum and going afterwards local minimum of cumulative returns.

    Parameters
    ----------
    portfolio_returns
        Portfolio, for maximum drawdown is calculated.
    """

    cumsum_returns = portfolio_returns.cumsum()
    return (cumsum_returns - cumsum_returns.cummax()).min()


def rolling_max_drawdown(portfolio_returns: pd.Series) -> pd.Series:
    """
    Calculates rolling on 1 trading year maximum drawdown of a `portfolio`.

    See function max_drawdown.

    Parameters
    ----------
    portfolio_returns
        Portfolio, for which maximum drawdown is calculated.
    """

    freq = _get_freq(portfolio_returns)
    cumsum_returns = portfolio_returns.cumsum()
    return cumsum_returns.rolling(freq).apply(lambda x: (x - x.cummax()).min())


def _get_freq(data: pd.Series | pd.DataFrame) -> int:
    """
    Extract inferred by pandas frequency of data to annualize metrics. Minimal
    supported frequency of data is daily, maximal - yearly.
    """

    freq_alias_to_num = {
        # yearly
        'BA': 1,
        'A': 1,
        # quarterly
        'BQ': 4,
        'Q': 4,
        # monthly
        'BM': 12,
        'M': 12,
        # weekly
        'W': 52,
        # daily
        'B': 252,
        'D': 252,
    }

    freq_num = freq_alias_to_num.get(data.index.inferred_freq)
    if freq_num is None:
        raise ValueError('periodicity of given data cannot be defined, '
                         'try to resample data')
    return freq_num


def _estimate_ols(
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
) -> sm_linear.RegressionResults:
    """
    Estimates simple linear regression (ordinary least squares) of portfolio
    returns per benchmark returns.
    """

    portfolio_returns = portfolio_returns
    start_trading = portfolio_returns.index[0]
    benchmark_returns = benchmark_returns[start_trading:]
    x = sm_tools.add_constant(benchmark_returns.values)
    return sm_linear.OLS(portfolio_returns.values, x).fit()


def _estimate_rolling_ols(
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
) -> sm_rolling.RollingRegressionResults:
    """
    Estimates rolling simple linear regression (ordinary least squares) of
    portfolio returns per benchmark returns. kek
    """

    portfolio_returns = portfolio_returns
    start_trading = portfolio_returns.index[0]
    benchmark_returns = benchmark_returns[start_trading:]
    x = sm_tools.add_constant(benchmark_returns.values)
    return sm_rolling.RollingOLS(portfolio_returns.values, x,
                                 window=_get_freq(portfolio_returns)).fit()
