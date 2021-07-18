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

# type alias for AbstractPortfolio | AbstractBenchmark
HasReturns = Union[pqr.portfolios.AbstractPortfolio,
                   pqr.benchmarks.AbstractBenchmark]


def summary(portfolio: HasReturns, benchmark: HasReturns) -> pd.Series:
    """
    Calculate summary statistics for one portfolio with respect to benchmark.

    Parameters
    ----------
    portfolio : AbstractPortfolio, AbstractBenchmark
        Portfolio, for which metrics are calculated.
    benchmark : AbstractPortfolio, AbstractBenchmark
        Benchmark, which used as the alternative for portfolio to calculate
        some metrics (usually some assets index).

    Returns
    -------
    pd.Series
        Series, where index represents different metrics and values are these
        metrics.
    """

    return pd.Series(
        {
            'Alpha, %': alpha(portfolio, benchmark) * 100,
            'Beta': beta(portfolio, benchmark),
            'Sharpe Ratio': sharpe(portfolio),
            'Mean Return, %': mean_return(portfolio) * 100,
            'Excessive Return, %': mean_excessive_return(portfolio,
                                                         benchmark) * 100,
            'Volatility, %': volatility(portfolio) * 100,
            'Benchmark Correlation': benchmark_correlation(portfolio,
                                                           benchmark),
            'Profitable Periods, %': profitable_periods_share(portfolio) * 100,
            'Maximum Drawdown, %': max_drawdown(portfolio) * 100,
        },
        name=repr(portfolio)
    ).round(2)


def alpha(portfolio: HasReturns, benchmark: HasReturns) -> Union[int, float]:
    """
    Calculates alpha of portfolio.

    Alpha is the coefficient :math:`\\alpha` in the estimated regression per
    benchmark returns:

    .. math:: r_{portfolio} = \\alpha + \\beta * r_{benchmark} + \\epsilon

    Parameters
    ----------
    portfolio : AbstractPortfolio, AbstractBenchmark
        Portfolio, for which alpha is calculated.
    benchmark : AbstractPortfolio, AbstractBenchmark
        Benchmark, which used as the alternative for portfolio.

    Returns
    -------
    int, float
        Value of alpha-coefficient for portfolio.
    """

    est = _estimate_ols(portfolio, benchmark)
    return est.params[0]


def rolling_alpha(portfolio: HasReturns, benchmark: HasReturns) -> pd.Series:
    """
    Calculates rolling on 1 trading year alpha of portfolio.

    See function alpha.

    Parameters
    ----------
    portfolio : AbstractPortfolio, AbstractBenchmark
        Portfolio, for which rolling alpha is calculated.
    benchmark : AbstractPortfolio, AbstractBenchmark
        Benchmark, which used as the alternative for portfolio.

    Returns
    -------
    pd.Series
        Series of rolling alpha-coefficient, doesn't return t-stats and
        p-values.
    """

    est = _estimate_rolling_ols(portfolio, benchmark)
    rolling_alpha_ = pd.Series(index=portfolio.returns.index)
    rolling_alpha_[portfolio.trading_start:] = est.params[:, 0]
    return rolling_alpha_


def beta(portfolio: HasReturns, benchmark: HasReturns) -> Union[int, float]:
    """
    Calculates beta of portfolio.

    Beta is the coefficient :math:`\\beta` in the estimated regression per
    benchmark returns:

    .. math:: r_{portfolio} = \\alpha + \\beta * r_{benchmark} + \\epsilon


    Parameters
    ----------
    portfolio : AbstractPortfolio, AbstractBenchmark
        Portfolio, for which beta is calculated.
    benchmark : AbstractPortfolio, AbstractBenchmark
        Benchmark, which used as the alternative for portfolio.

    Returns
    -------
    int, float
        Value of beta-coefficient for portfolio.
    """

    est = _estimate_ols(portfolio, benchmark)
    return est.params[1]


def rolling_beta(portfolio: HasReturns, benchmark: HasReturns) -> pd.Series:
    """
    Calculates rolling on 1 trading year beta of portfolio.

    See function beta.

    Parameters
    ----------
    portfolio : AbstractPortfolio, AbstractBenchmark
        Portfolio, for which rolling beta is calculated.
    benchmark : AbstractPortfolio, AbstractBenchmark
        Benchmark, which used as the alternative for portfolio.

    Returns
    -------
    pd.Series
        Series of rolling beta-coefficient, doesn't return t-stats and
        p-values.
    """

    est = _estimate_rolling_ols(portfolio, benchmark)
    rolling_beta_ = pd.Series(index=portfolio.returns.index)
    rolling_beta_[portfolio.trading_start:] = est.params[:, 1]
    return rolling_beta_


def sharpe(portfolio: HasReturns) -> Union[int, float]:
    """
    Calculates sharpe ratio of portfolio.

    Sharpe Ratio is the ratio of mean return and standard deviation of
    returns:

    .. math:: SR = \\frac{\\bar{r}}{\\sigma_{r}}

    Parameters
    ----------
    portfolio : AbstractPortfolio, AbstractBenchmark
        Portfolio, for which sharpe ratio is calculated.

    Returns
    -------
    int, float
        Value of annualized sharpe ratio for portfolio.
    """

    annualization_rate = np.sqrt(_get_freq(portfolio.returns))
    return mean_return(portfolio) / volatility(portfolio) * annualization_rate


def rolling_sharpe(portfolio: HasReturns) -> pd.Series:
    """
    Calculates rolling on 1 trading year sharpe ratio of portfolio.

    See function sharpe.

    Parameters
    ----------
    portfolio : AbstractPortfolio, AbstractBenchmark
        Portfolio, for which rolling sharpe ratio is calculated.

    Returns
    -------
    pd.Series
        Series of rolling sharpe ratio.
    """

    annualization_rate = np.sqrt(_get_freq(portfolio.returns))
    mean = rolling_mean_return(portfolio)
    std = rolling_volatility(portfolio)
    return mean / std * annualization_rate


def mean_return(portfolio: HasReturns) -> Union[int, float]:
    """
    Calculates mean return of portfolio.

    Mean Return is simple expected value:

    .. math:: \\bar{r} = \\frac{\\sum_{i=1}^{n}r}{n}

    Parameters
    ----------
    portfolio : AbstractPortfolio, AbstractBenchmark
        Portfolio, for which mean return is calculated.

    Returns
    -------
    int, float
        Value of mean return for portfolio.
    """

    return portfolio.returns.mean()


def rolling_mean_return(portfolio: HasReturns) -> pd.Series:
    """
    Calculates rolling on 1 trading year mean return of portfolio.

    See function mean_return.

    Parameters
    ----------
    portfolio : AbstractPortfolio, AbstractBenchmark
        Portfolio, for which mean return is calculated.

    Returns
    -------
    pd.Series
        Series of rolling mean return.
    """

    freq = _get_freq(portfolio.returns)
    return portfolio.returns.rolling(freq).mean()


def mean_excessive_return(portfolio: HasReturns,
                          benchmark: HasReturns) -> Union[int, float]:
    """
    Calculates mean excessive return of portfolio.

    Mean Excessive Return is difference between mean return of portfolio and
    mean return of benchmark:

    .. math:: MER = \\overline{r_{portfolio}} - \\overline{r_{benchmark}}

    Parameters
    ----------
    portfolio : AbstractPortfolio, AbstractBenchmark
        Portfolio, for which alpha is calculated.
    benchmark : AbstractPortfolio, AbstractBenchmark
        Benchmark, which used as the alternative for portfolio.

    Returns
    -------
    int, float
        Value of mean excessive return of portfolio.
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
    portfolio : AbstractPortfolio, AbstractBenchmark
        Portfolio, for which rolling mean excessive return is calculated.
    benchmark : AbstractPortfolio, AbstractBenchmark
        Benchmark, which used as the alternative for portfolio.

    Returns
    -------
    pd.Series
        Series of rolling mean excessive return.
    """

    mean_portfolio_return = rolling_mean_return(portfolio)
    mean_benchmark_return = rolling_mean_return(benchmark)
    return mean_portfolio_return - mean_benchmark_return


def volatility(portfolio: HasReturns) -> Union[int, float]:
    """
    Calculates volatility of return of a portfolio.

    Volatility of the portfolio is standard deviation of portfolio returns:

    .. math:: \\sigma_r=\\sqrt{\\frac{\\sum_{i=1}^{n}(r_i-\\bar{r})^2}{n-1}}

    Parameters
    ----------
    portfolio : AbstractPortfolio, AbstractBenchmark
        Portfolio, for which volatility is calculated.

    Returns
    -------
    int, float
        Value of volatility for portfolio.
    """

    return portfolio.returns.std()


def rolling_volatility(portfolio: HasReturns) -> pd.Series:
    """
    Calculates rolling on 1 trading year volatility of portfolio.

    See function volatility.

    Parameters
    ----------
    portfolio : AbstractPortfolio, AbstractBenchmark
        Portfolio, for which volatility is calculated.

    Returns
    -------
    pd.Series
        Series of rolling volatility.
    """

    freq = _get_freq(portfolio.returns)
    return portfolio.returns.rolling(freq).std()


def benchmark_correlation(portfolio: HasReturns,
                          benchmark: HasReturns) -> Union[int, float]:
    """
    Calculates correlation between portfolio and benchmark.

    Benchmark Correlation is simple correlation between 2 time-series - returns
    of portfolio and returns of benchmark:

    .. math:: BC = corr(r_{portfolio}, r_{benchmark})

    Parameters
    ----------
    portfolio : AbstractPortfolio, AbstractBenchmark
        Portfolio, for which correlation with benchmark is calculated.
    benchmark : AbstractPortfolio, AbstractBenchmark
        Benchmark, which used as the alternative for portfolio.

    Returns
    -------
    int, float
        Value of correlation between benchmark and portfolio.
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
    portfolio : AbstractPortfolio, AbstractBenchmark
        Portfolio, for which rolling correlation between portfolio and
        benchmark is calculated.
    benchmark : AbstractPortfolio, AbstractBenchmark
        Benchmark, which used as the alternative for portfolio.

    Returns
    -------
    pd.Series
        Series of rolling correlation between portfolio and benchmark.
    """

    freq = _get_freq(portfolio.returns)
    return portfolio.returns.rolling(freq).corr(benchmark.returns)


def profitable_periods_share(portfolio: HasReturns) -> Union[int, float]:
    """
    Calculates share of profitable periods of portfolio.

    Share of Profitable Periods of portfolios is simple ratio of number of
    periods with positive returns and total number of trading periods:

    .. math:: SPR = \\frac{\\sum_{i=1}^{n}[r_{i} > 0]}{n}

    Parameters
    ----------
    portfolio : AbstractPortfolio, AbstractBenchmark
        Portfolio, for which share of profitable periods is calculated.

    Returns
    -------
    int, float
        Value of share of profitable periods for portfolio.
    """

    positive_periods = (portfolio.returns > 0).sum()
    total_periods = np.size(portfolio.returns)
    return positive_periods / total_periods


def rolling_profitable_periods_share(portfolio: HasReturns) -> pd.Series:
    """
    Calculates rolling on 1 trading year share of profitable periods of
    portfolio.

    See function profitable_periods_share.

    Parameters
    ----------
    portfolio : AbstractPortfolio, AbstractBenchmark
        Portfolio, for which share of profitable periods is calculated.

    Returns
    -------
    pd.Series
        Series of rolling share of profitable periods.
    """

    freq = _get_freq(portfolio.returns)
    rolling_positive = (portfolio.returns > 0).rolling(freq).sum()
    return rolling_positive / freq


def max_drawdown(portfolio: HasReturns) -> Union[int, float]:
    """
    Calculates maximum drawdown of portfolio.

    Maximum Drawdown of portfolio is the biggest difference between local
    maximum and going afterwards local minimum of cumulative returns:

    .. math:: MD = min\{cumsum(r) - cummax(cumsum(r))\}

    Parameters
    ----------
    portfolio : AbstractPortfolio, AbstractBenchmark
        Portfolio, for maximum drawdown is calculated.

    Returns
    -------
    int, float
        Value of maximum drawdown for portfolio.
    """

    cumsum_returns = portfolio.returns.cumsum()
    return (cumsum_returns - cumsum_returns.cummax()).min()


def rolling_max_drawdown(portfolio: HasReturns) -> pd.Series:
    """
    Calculates rolling on 1 trading year maximum drawdown of portfolio.

    See function max_drawdown.

    Parameters
    ----------
    portfolio : AbstractPortfolio, AbstractBenchmark
        Portfolio, for which maximum drawdown is calculated.

    Returns
    -------
    pd.Series
        Series of rolling maximum drawdown.
    """

    freq = _get_freq(portfolio.returns)
    cumsum_returns = portfolio.returns.cumsum()
    return cumsum_returns.rolling(freq).apply(lambda x: (x - x.cummax()).min())


def _get_freq(data: Union[pd.Series, pd.DataFrame]) -> int:
    """
    Extract inferred by pandas frequency of data to annualize metrics.

    Parameters
    ----------
    data : pd.DataFrame, pd.Series

    Returns
    -------
    int
        Number of trading periods in 1 year (e.g. if inferred_freq is "BM" or
        "M", than function returns 12, because there are 12 trading months in
        1 year).

    Notes
    -----
    Minimal supported frequency of data is daily, maximal - yearly.
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

    Parameters
    ----------
    portfolio : AbstractPortfolio, AbstractBenchmark
        Portfolio, which returns are used for regression per benchmark returns.
    benchmark : AbstractPortfolio, AbstractBenchmark
        Benchmark, which returns are used for regression as the basis for
        portfolio returns.

    Returns
    -------
    sm_linear.RegressionResults
        Object, containing results of linear regression.
    """

    portfolio_returns = portfolio.returns[portfolio.trading_start:]
    benchmark_returns = benchmark.returns[portfolio.trading_start:]
    x = sm_tools.add_constant(benchmark_returns.values)
    return sm_linear.OLS(portfolio_returns.values, x).fit()


def _estimate_rolling_ols(
        portfolio: HasReturns,
        benchmark: HasReturns) -> sm_rolling.RollingRegressionResults:
    """
    Estimates rolling simple linear regression (ordinary least squares) of
    portfolio returns per benchmark returns.

    Parameters
    ----------
    portfolio : AbstractPortfolio, AbstractBenchmark
        Portfolio, which returns are used for regression per benchmark returns.
    benchmark : AbstractPortfolio, AbstractBenchmark
        Benchmark, which returns are used for regression as the basis for
        portfolio returns.

    Returns
    -------
    sm_linear.RollingRegressionResults
        Object, containing results of rolling linear regression.
    """

    portfolio_returns = portfolio.returns[portfolio.trading_start:]
    benchmark_returns = benchmark.returns[portfolio.trading_start:]
    x = sm_tools.add_constant(benchmark_returns.values)
    return sm_rolling.RollingOLS(portfolio_returns.values, x,
                                 window=_get_freq(portfolio_returns)).fit()
