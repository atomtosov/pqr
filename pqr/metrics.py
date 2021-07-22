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


from typing import Union

import numpy as np
import pandas as pd
import statsmodels.regression.linear_model as sm_linear
import statsmodels.regression.rolling as sm_rolling
import statsmodels.tools.tools as sm_tools

import pqr.benchmarks
import pqr.portfolios


__all__ = [
    'summary',
    'alpha', 'rolling_alpha',
    'beta', 'rolling_beta',
    'sharpe', 'rolling_sharpe',
    'mean_return', 'rolling_mean_return',
    'mean_excessive_return', 'rolling_mean_excessive_return',
    'volatility', 'rolling_volatility',
    'benchmark_correlation', 'rolling_benchmark_correlation',
    'profitable_periods_share', 'rolling_profitable_periods_share',
    'max_drawdown', 'rolling_max_drawdown',
]

# type alias for Portfolio | Benchmark
HasReturns = Union[pqr.portfolios.Portfolio,
                   pqr.benchmarks.Benchmark]


def summary(portfolio: HasReturns, benchmark: HasReturns) -> pd.Series:
    """
    Calculates summary statistics for a `portfolio`.

    Computed metrics:

    - Alpha, %
    - Beta, %
    - Sharpe Ratio
    - Mean Return, %
    - Mean Excessive Return, %
    - Volatility, %
    - Benchmark Correlation, %
    - Profitable periods, %
    - Maximum Drawdown, %

    Parameters
    ----------
    portfolio
        Portfolio, for which metrics are calculated.
    benchmark
        Benchmark, which used as the alternative for the `portfolio` to
        calculate some metrics (usually some stock index).
    """

    return pd.Series(
        {
            'Alpha, %': alpha(portfolio, benchmark) * 100,
            'Beta': beta(portfolio, benchmark),
            'Sharpe Ratio': sharpe(portfolio),
            'Mean Return, %': mean_return(portfolio) * 100,
            'Mean Excessive Return, %': mean_excessive_return(portfolio,
                                                              benchmark) * 100,
            'Volatility, %': volatility(portfolio) * 100,
            'Benchmark Correlation': benchmark_correlation(portfolio,
                                                           benchmark),
            'Profitable Periods, %': profitable_periods_share(portfolio) * 100,
            'Maximum Drawdown, %': max_drawdown(portfolio) * 100,
        },
        name=str(portfolio)
    ).round(2)


def alpha(portfolio: HasReturns, benchmark: HasReturns) -> Union[int, float]:
    """
    Calculates alpha of a `portfolio`.

    Alpha is the coefficient :math:`\\alpha` in the estimated regression per
    benchmark returns:

    .. math:: r_{portfolio} = \\alpha + \\beta * r_{benchmark} + \\epsilon

    Parameters
    ----------
    portfolio
        Portfolio, for which alpha is calculated.
    benchmark
        Benchmark, which used as the alternative for the `portfolio`.
    """

    est = _estimate_ols(portfolio, benchmark)
    return est.params[0]


def rolling_alpha(portfolio: HasReturns, benchmark: HasReturns) -> pd.Series:
    """
    Calculates rolling on 1 trading year alpha of a `portfolio`.

    See function alpha.

    Parameters
    ----------
    portfolio
        Portfolio, for which rolling alpha is calculated.
    benchmark
        Benchmark, which used as the alternative for the `portfolio`.
    """

    est = _estimate_rolling_ols(portfolio, benchmark)
    return pd.Series(est.params[:, 0], index=portfolio.returns.index)


def beta(portfolio: HasReturns, benchmark: HasReturns) -> Union[int, float]:
    """
    Calculates beta of a `portfolio`.

    Beta is the coefficient :math:`\\beta` in the estimated regression per
    benchmark returns:

    .. math:: r_{portfolio} = \\alpha + \\beta * r_{benchmark} + \\epsilon


    Parameters
    ----------
    portfolio
        Portfolio, for which beta is calculated.
    benchmark
        Benchmark, which used as the alternative for the `portfolio`.
    """

    est = _estimate_ols(portfolio, benchmark)
    return est.params[1]


def rolling_beta(portfolio: HasReturns, benchmark: HasReturns) -> pd.Series:
    """
    Calculates rolling on 1 trading year beta of a `portfolio`.

    See function beta.

    Parameters
    ----------
    portfolio
        Portfolio, for which rolling beta is calculated.
    benchmark
        Benchmark, which used as the alternative for the `portfolio`.
    """

    est = _estimate_rolling_ols(portfolio, benchmark)
    return pd.Series(est.params[:, 1], index=portfolio.returns.index)


def sharpe(portfolio: HasReturns) -> Union[int, float]:
    """
    Calculates sharpe ratio of a `portfolio`.

    Sharpe Ratio is the ratio of mean return and standard deviation of
    returns:

    .. math:: SR = \\frac{\\bar{r}}{\\sigma_{r}}

    Parameters
    ----------
    portfolio
        Portfolio, for which sharpe ratio is calculated.
    """

    annualization_rate = np.sqrt(_get_freq(portfolio.returns))
    return mean_return(portfolio) / volatility(portfolio) * annualization_rate


def rolling_sharpe(portfolio: HasReturns) -> pd.Series:
    """
    Calculates rolling on 1 trading year sharpe ratio of a `portfolio`.

    See function sharpe.

    Parameters
    ----------
    portfolio
        Portfolio, for which rolling sharpe ratio is calculated.
    """

    annualization_rate = np.sqrt(_get_freq(portfolio.returns))
    mean = rolling_mean_return(portfolio)
    std = rolling_volatility(portfolio)
    return mean / std * annualization_rate


def mean_return(portfolio: HasReturns) -> Union[int, float]:
    """
    Calculates mean return of a `portfolio`.

    Mean Return is simple expected value:

    .. math:: \\bar{r} = \\frac{\\sum_{i=1}^{n}r}{n}

    Parameters
    ----------
    portfolio
        Portfolio, for which mean return is calculated.
    """

    return portfolio.returns.mean()


def rolling_mean_return(portfolio: HasReturns) -> pd.Series:
    """
    Calculates rolling on 1 trading year mean return of a `portfolio`.

    See function mean_return.

    Parameters
    ----------
    portfolio
        Portfolio, for which mean return is calculated.
    """

    freq = _get_freq(portfolio.returns)
    return portfolio.returns.rolling(freq).mean()


def mean_excessive_return(portfolio: HasReturns,
                          benchmark: HasReturns) -> Union[int, float]:
    """
    Calculates mean excessive return of a `portfolio`.

    Mean Excessive Return is difference between mean return of portfolio and
    mean return of benchmark:

    .. math:: MER = \\overline{r_{portfolio}} - \\overline{r_{benchmark}}

    Parameters
    ----------
    portfolio
        Portfolio, for which alpha is calculated.
    benchmark
        Benchmark, which used as the alternative for the `portfolio`.
    """

    mean_portfolio_return = mean_return(portfolio)
    mean_benchmark_return = mean_return(benchmark)
    return mean_portfolio_return - mean_benchmark_return


def rolling_mean_excessive_return(portfolio: HasReturns,
                                  benchmark: HasReturns) -> pd.Series:
    """
    Calculates rolling on 1 trading year mean excessive return of portfolio.

    See function mean_excessive_return.

    Parameters
    ----------
    portfolio
        Portfolio, for which rolling mean excessive return is calculated.
    benchmark
        Benchmark, which used as the alternative for the `portfolio`.
    """

    mean_portfolio_return = rolling_mean_return(portfolio)
    mean_benchmark_return = rolling_mean_return(benchmark)
    return mean_portfolio_return - mean_benchmark_return


def volatility(portfolio: HasReturns) -> Union[int, float]:
    """
    Calculates volatility of returns of a `portfolio`.

    Volatility of the portfolio is standard deviation of portfolio returns:

    .. math:: \\sigma_r=\\sqrt{\\frac{\\sum_{i=1}^{n}(r_i-\\bar{r})^2}{n-1}}

    Parameters
    ----------
    portfolio
        Portfolio, for which volatility is calculated.
    """

    return portfolio.returns.std()


def rolling_volatility(portfolio: HasReturns) -> pd.Series:
    """
    Calculates rolling on 1 trading year volatility of a `portfolio`.

    See function volatility.

    Parameters
    ----------
    portfolio
        Portfolio, for which volatility is calculated.
    """

    freq = _get_freq(portfolio.returns)
    return portfolio.returns.rolling(freq).std()


def benchmark_correlation(portfolio: HasReturns,
                          benchmark: HasReturns) -> Union[int, float]:
    """
    Calculates correlation between a `portfolio` and a `benchmark`.

    Benchmark Correlation is simple correlation between 2 time-series - returns
    of portfolio and returns of benchmark:

    .. math:: BC = corr(r_{portfolio}, r_{benchmark})

    Parameters
    ----------
    portfolio
        Portfolio, for which correlation with the `benchmark` is calculated.
    benchmark
        Benchmark, which used as the alternative for the `portfolio`.
    """

    return portfolio.returns.corr(benchmark.returns)


def rolling_benchmark_correlation(portfolio: HasReturns,
                                  benchmark: HasReturns) -> pd.Series:
    """
    Calculates rolling on 1 trading year correlation between portfolio and
    benchmark.

    See function benchmark_correlation.

    Parameters
    ----------
    portfolio
        Portfolio, for which rolling correlation with the `benchmark` is
        calculated.
    benchmark
        Benchmark, which used as the alternative for the `portfolio`.
    """

    freq = _get_freq(portfolio.returns)
    return portfolio.returns.rolling(freq).corr(benchmark.returns)


def profitable_periods_share(portfolio: HasReturns) -> Union[int, float]:
    """
    Calculates share of profitable periods of a `portfolio`.

    Share of Profitable Periods of portfolios is simple ratio of number of
    periods with positive returns and total number of trading periods:

    .. math:: SPR = \\frac{\\sum_{i=1}^{n}[r_{i} > 0]}{n}

    Parameters
    ----------
    portfolio
        Portfolio, for which share of profitable periods is calculated.
    """

    positive_periods = (portfolio.returns > 0).sum()
    total_periods = np.size(portfolio.returns)
    return positive_periods / total_periods


def rolling_profitable_periods_share(portfolio: HasReturns) -> pd.Series:
    """
    Calculates rolling on 1 trading year share of profitable periods of a
    `portfolio`.

    See function profitable_periods_share.

    Parameters
    ----------
    portfolio
        Portfolio, for which share of profitable periods is calculated.
    """

    freq = _get_freq(portfolio.returns)
    rolling_positive = (portfolio.returns > 0).rolling(freq).sum()
    return rolling_positive / freq


def max_drawdown(portfolio: HasReturns) -> Union[int, float]:
    """
    Calculates maximum drawdown of a `portfolio`.

    Maximum Drawdown of portfolio is the highest relative difference between
    local maximum and going afterwards local minimum of cumulative returns.

    Parameters
    ----------
    portfolio
        Portfolio, for maximum drawdown is calculated.
    """

    cumsum_returns = portfolio.returns.cumsum()
    return (cumsum_returns - cumsum_returns.cummax()).min()


def rolling_max_drawdown(portfolio: HasReturns) -> pd.Series:
    """
    Calculates rolling on 1 trading year maximum drawdown of a `portfolio`.

    See function max_drawdown.

    Parameters
    ----------
    portfolio
        Portfolio, for which maximum drawdown is calculated.
    """

    freq = _get_freq(portfolio.returns)
    cumsum_returns = portfolio.returns.cumsum()
    return cumsum_returns.rolling(freq).apply(lambda x: (x - x.cummax()).min())


def _get_freq(data: Union[pd.Series, pd.DataFrame]) -> int:
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


def _estimate_ols(portfolio: HasReturns,
                  benchmark: HasReturns) -> sm_linear.RegressionResults:
    """
    Estimates simple linear regression (ordinary least squares) of portfolio
    returns per benchmark returns.
    """

    portfolio_returns = portfolio.returns
    start_trading = portfolio.returns.index[0]
    benchmark_returns = benchmark.returns[start_trading:]
    x = sm_tools.add_constant(benchmark_returns.values)
    return sm_linear.OLS(portfolio_returns.values, x).fit()


def _estimate_rolling_ols(
        portfolio: HasReturns,
        benchmark: HasReturns) -> sm_rolling.RollingRegressionResults:
    """
    Estimates rolling simple linear regression (ordinary least squares) of
    portfolio returns per benchmark returns. kek
    """

    portfolio_returns = portfolio.returns
    start_trading = portfolio.returns.index[0]
    benchmark_returns = benchmark.returns[start_trading:]
    x = sm_tools.add_constant(benchmark_returns.values)
    return sm_rolling.RollingOLS(portfolio_returns.values, x,
                                 window=_get_freq(portfolio_returns)).fit()
