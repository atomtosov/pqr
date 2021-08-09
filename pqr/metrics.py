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

from typing import Callable, Optional

import numpy as np
import pandas as pd
import statsmodels.regression.linear_model as sm_linear
import statsmodels.tools.tools as sm_tools

import pqr.benchmarks
import pqr.portfolios

__all__ = [
    'summary',

    'cumulative_returns', 'total_return',
    'annual_return', 'annual_volatility',

    'mean_return', 'rolling_mean_return',
    'volatility', 'rolling_volatility',
    'max_drawdown', 'rolling_max_drawdown',
    'win_rate', 'rolling_win_rate',

    'value_at_risk', 'rolling_value_at_risk',
    'expected_tail_loss', 'rolling_expected_tail_loss',
    'rachev_ratio', 'rolling_rachev_ratio',

    'calmar_ratio', 'rolling_calmar_ratio',
    'sharpe_ratio', 'rolling_sharpe_ratio',
    'omega_ratio', 'rolling_omega_ratio',
    'sortino_ratio', 'rolling_sortino_ratio',
    'downside_risk', 'rolling_downside_risk',

    'mean_excess_return', 'rolling_mean_excess_return',
    'benchmark_correlation', 'rolling_benchmark_correlation',
    'alpha', 'rolling_alpha',
    'beta', 'rolling_beta',
]


def summary(
        portfolio: pqr.portfolios.AbstractPortfolio,
        benchmark: pqr.benchmarks.Benchmark
) -> pd.Series:
    """
    Calculates summary statistics for a `portfolio`.

    Computed metrics:

    - Alpha, %
    - Beta, %
    - Sharpe Ratio
    - Mean Return, %
    - Volatility, %
    - Excess Return, %
    - Benchmark Correlation, %
    - Win Rate, %
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
            'Sharpe Ratio': sharpe_ratio(portfolio),
            'Mean Return, %': mean_return(portfolio) * 100,
            'Volatility, %': volatility(portfolio) * 100,
            'Mean Excess Return, %': mean_excess_return(portfolio,
                                                        benchmark) * 100,
            'Benchmark Correlation': benchmark_correlation(portfolio,
                                                           benchmark),
            'Win Rate, %': win_rate(portfolio) * 100,
            'Maximum Drawdown, %': max_drawdown(portfolio) * 100,
        },
        name=portfolio.name
    ).round(2)


def cumulative_returns(
        portfolio: pqr.portfolios.AbstractPortfolio | pqr.benchmarks.Benchmark
) -> pd.Series:
    """
    Calculates Cumulative Returns of a `portfolio`.

    Parameters
    ----------
    portfolio
        An allocated portfolio or a benchmark.

    Notes
    -----
    Starting point equals to zero.
    """

    return _cumulative_returns(portfolio.returns)


def total_return(portfolio: pqr.portfolios.AbstractPortfolio) -> int | float:
    """
    Calculates Total Return of a `portfolio`.

    Parameters
    ----------
    portfolio
        An allocated portfolio.

    Notes
    -----
    Shows additional return, not equity curve's state.
    """

    return _total_return(portfolio.returns)


def annual_return(portfolio: pqr.portfolios.AbstractPortfolio) -> int | float:
    """
    Calculates Annual Return of a `portfolio` as CAGR.

    CAGR (Compounded Annual Growth Rate) calculated as:

    .. math::
        CAGR = (1 + Total Return)^{\\frac{1}{Years}} - 1

    Parameters
    ----------
    portfolio
        An allocated portfolio.
    """

    return _annual_return(portfolio.returns)


def annual_volatility(
        portfolio: pqr.portfolios.AbstractPortfolio) -> int | float:
    """
    Calculates Annual Volatility of returns of a `portfolio`.

    Annual Volatility of the portfolio is the annualized standard deviation of
    portfolio returns:

    .. math::
        \\sigma_r * \\sqrt{Number\;of\;Periods\;in\;a\;Year}

    Parameters
    ----------
    portfolio
        An allocated portfolio.
    """

    return _annual_volatility(portfolio.returns)


def mean_return(portfolio: pqr.portfolios.AbstractPortfolio) -> int | float:
    """
    Calculates Mean Return of a `portfolio`.

    Mean Return is simple expected value:

    .. math::
        E(r) = \\bar{r}

    Parameters
    ----------
    portfolio
        An allocated portfolio.
    """

    return _mean_return(portfolio.returns)


def rolling_mean_return(
        portfolio: pqr.portfolios.AbstractPortfolio,
        window: Optional[int] = None
) -> pd.Series:
    """
    Calculates rolling Mean Return of a `portfolio`.

    See :func:`~pqr.metrics.mean_return`.

    Parameters
    ----------
    portfolio
        An allocated portfolio.
    window
        Number of observations on a rolling window. If not passed, `window`
        equals to approximate number of periods in a year.
    """

    return _roll(
        portfolio.returns,
        metric=_mean_return,
        window=window
    )


def volatility(portfolio: pqr.portfolios.AbstractPortfolio) -> int | float:
    """
    Calculates Volatility of returns of a `portfolio`.

    Volatility of the portfolio is the standard deviation of portfolio returns:

    .. math::
        \\sigma_r = \\sqrt{\\frac{\\sum_{i=1}^{n}(r_i-\\bar{r})^2}{n-1}}

    Parameters
    ----------
    portfolio
        An allocated portfolio.
    """

    return _volatility(portfolio.returns)


def rolling_volatility(
        portfolio: pqr.portfolios.AbstractPortfolio,
        window: Optional[int] = None
) -> pd.Series:
    """
    Calculates rolling Volatility of a `portfolio`.

    See :func:`~pqr.metrics.volatility`.

    Parameters
    ----------
    portfolio
        An allocated portfolio.
    window
        Number of observations on a rolling window. If not passed, `window`
        equals to approximate number of periods in a year.
    """

    return _roll(
        portfolio.returns,
        metric=_volatility,
        window=window
    )


def max_drawdown(portfolio: pqr.portfolios.AbstractPortfolio) -> int | float:
    """
    Calculates Maximum Drawdown of a `portfolio`.

    Maximum Drawdown of portfolio is the highest relative difference between
    high water mark (cumulative maximum of the cumulative returns) and
    cumulative returns:

    .. math::
        MDD = -\max{\\frac{HWM - Cumulative\;Returns}{HWM}}

    Parameters
    ----------
    portfolio
        An allocated portfolio.
    """

    return _max_drawdown(portfolio.returns)


def rolling_max_drawdown(
        portfolio: pqr.portfolios.AbstractPortfolio,
        window: Optional[int] = None
) -> pd.Series:
    """
    Calculates rolling Maximum Drawdown of a `portfolio`.

    See :func:`~pqr.metrics.max_drawdown`.

    Parameters
    ----------
    portfolio
        An allocated portfolio.
    window
        Number of observations on a rolling window. If not passed, `window`
        equals to approximate number of periods in a year.
    """

    return _roll(
        portfolio.returns,
        metric=_max_drawdown,
        window=window
    )


def win_rate(portfolio: pqr.portfolios.AbstractPortfolio) -> int | float:
    """
    Calculates Win Rate of a `portfolio`.

    Win Rate of a portfolio is simple ratio of number of periods with positive
    returns and total number of trading periods:

    .. math::
        WR = \\frac{\\sum_{i=1}^{n}[r_{i} > 0]}{n}

    Parameters
    ----------
    portfolio
        An allocated portfolio.
    """

    return _win_rate(portfolio.returns)


def rolling_win_rate(
        portfolio: pqr.portfolios.AbstractPortfolio,
        window: Optional[int] = None
) -> pd.Series:
    """
    Calculates rolling Win Rate of a `portfolio`.

    See :func:`~pqr.metrics.win_rate`.

    Parameters
    ----------
    portfolio
        An allocated portfolio.
    window
        Number of observations on a rolling window. If not passed, `window`
        equals to approximate number of periods in a year.
    """

    return _roll(
        portfolio.returns,
        metric=_win_rate,
        window=window
    )


def value_at_risk(
        portfolio: pqr.portfolios.AbstractPortfolio,
        confidence_level: int | float = 0.95
) -> int | float:
    """
    Calculates Value at Risk of a `portfolio`.

    VaR shows the amount of potential loss that could happen in a portfolio
    with given `confidence_level`:

    .. math::
        VaR = -\\inf\{F_r(r) > Confidence\;Level\}

    Parameters
    ----------
    portfolio
        An allocated portfolio.
    confidence_level
        The probability, with which the estimation of VaR is true.
    """

    return _value_at_risk(portfolio.returns, confidence_level)


def rolling_value_at_risk(
        portfolio: pqr.portfolios.AbstractPortfolio,
        confidence_level: int | float = 0.95,
        window: Optional[int] = None
) -> pd.Series:
    """
    Calculates rolling Value at Risk of a `portfolio`.

    See :func:`~pqr.metrics.value_at_risk`.

    Parameters
    ----------
    portfolio
        An allocated portfolio.
    confidence_level
        The probability, with which the estimation of VaR is true.
    window
        Number of observations on a rolling window. If not passed, `window`
        equals to approximate number of periods in a year.
    """

    return _roll(
        portfolio.returns,
        metric=_value_at_risk,
        window=window,
        confidence_level=confidence_level
    )


def expected_tail_loss(
        portfolio: pqr.portfolios.AbstractPortfolio,
        confidence_level: int | float = 0.95
) -> int | float:
    """
    Calculates Expected Tail Loss of a `portfolio`.

    Expected Tail Loss shows the average of the values that fall beyond the
    VaR, calculated with `confidence_level`:

    .. math::
        ETL = \\frac{\\sum_{i=1}^{n}r_i\cdot[r_i \le VaR]}
        {\\sum_{i=1}^{n}[r_i \le VaR]}

    Parameters
    ----------
    portfolio
        An allocated portfolio.
    confidence_level
        The probability, with which the estimation of VaR is true.
    """

    return _expected_tail_loss(portfolio.returns, confidence_level)


def rolling_expected_tail_loss(
        portfolio: pqr.portfolios.AbstractPortfolio,
        confidence_level: int | float = 0.95,
        window: Optional[int] = None
) -> pd.Series:
    """
    Calculates rolling Expected Tail Loss of a `portfolio`.

    See :func:`~pqr.metrics.expected_tail_loss`.

    Parameters
    ----------
    portfolio
        An allocated portfolio.
    confidence_level
        The probability, with which the estimation of VaR is true.
    window
        Number of observations on a rolling window. If not passed, `window`
        equals to approximate number of periods in a year.
    """

    return _roll(
        portfolio.returns,
        metric=_expected_tail_loss,
        window=window,
        confidence_level=confidence_level
    )


def rachev_ratio(
        portfolio: pqr.portfolios.AbstractPortfolio,
        reward_cutoff: int | float = 0.95,
        risk_cutoff: int | float = 0.05,
        risk_free_rate: int | float | pd.Series = 0
) -> int | float:
    """
    Calculates Rachev Ratio (R-Ratio) of a `portfolio`.

    Rachev Ratio calculated as ETR (Expected Tail Return) divided by ETL
    (Expected Tail Return) of adjusted by `risk_free_rate` returns:

    .. math::
        R-Ratio = \\frac{ETR(r)_\\alpha}{ETL(r)_\\beta}

    Parameters
    ----------
    portfolio
        An allocated portfolio.
    reward_cutoff
        Cutoff to calculate expected tail return.
    risk_cutoff
        Cutoff to calculate expected tail loss. Confidence level to compute it
        equals to 1 - `risk_cutoff`.
    risk_free_rate
        Indicative rate of guaranteed returns (e.g. US government bond rate).
    """

    return _rachev_ratio(portfolio.returns, reward_cutoff, risk_cutoff,
                         risk_free_rate)


def rolling_rachev_ratio(
        portfolio: pqr.portfolios.AbstractPortfolio,
        reward_cutoff: int | float = 0.95,
        risk_cutoff: int | float = 0.05,
        risk_free_rate: int | float | pd.Series = 0,
        window: Optional[int] = None
) -> pd.Series:
    """
    Calculates rolling Rachev Ratio (R-Ratio) of a `portfolio`.

    See :func:`~pqr.metrics.rachev_ratio`.

    Parameters
    ----------
    portfolio
        An allocated portfolio.
    reward_cutoff
        Cutoff to calculate expected tail return.
    risk_cutoff
        Cutoff to calculate expected tail loss. Confidence level to compute it
        equals to 1 - `risk_cutoff`.
    risk_free_rate
        Indicative rate of guaranteed returns (e.g. US government bond rate).
    window
        Number of observations on a rolling window. If not passed, `window`
        equals to approximate number of periods in a year.
    """

    return _roll(
        portfolio.returns,
        metric=_mean_return,
        window=window,
        reward_cutoff=reward_cutoff,
        risk_cutoff=risk_cutoff,
        risk_free_rate=risk_free_rate
    )


def calmar_ratio(portfolio: pqr.portfolios.AbstractPortfolio) -> int | float:
    """
    Calculates Calmar Ratio of a `portfolio`.

    Calmar Ratio is annual return (CAGR) divided by maximum drawdown of the
    period:

    .. math::
        CR = \\frac{CAGR(r)}{MDD(r)}

    Parameters
    ----------
    portfolio
        An allocated portfolio.
    """

    return _calmar_ratio(portfolio.returns)


def rolling_calmar_ratio(
        portfolio: pqr.portfolios.AbstractPortfolio,
        window: Optional[int] = None
) -> pd.Series:
    """
    Calculates rolling Calmar Ratio of a `portfolio`.

    See :func:`~pqr.metrics.calmar_ratio`.

    Parameters
    ----------
    portfolio
        An allocated portfolio.
    window
        Number of observations on a rolling window. If not passed, `window`
        equals to approximate number of periods in a year.
    """

    return _roll(
        portfolio.returns,
        metric=_calmar_ratio,
        window=window
    )


def sharpe_ratio(
        portfolio: pqr.portfolios.AbstractPortfolio,
        risk_free_rate: int | float | pd.Series = 0
) -> int | float:
    """
    Calculates Sharpe Ratio of a `portfolio`.

    Sharpe Ratio calculated as annualized ratio between mean and volatility of
    adjusted by `risk_free_rate` returns:

    .. math::
        SR = \\frac{\\bar{r}}{\\sigma_{r}}

    Parameters
    ----------
    portfolio
        An allocated portfolio.
    risk_free_rate
        Indicative rate of guaranteed returns (e.g. US government bond rate).
    """

    return _sharpe_ratio(portfolio.returns, risk_free_rate)


def rolling_sharpe_ratio(
        portfolio: pqr.portfolios.AbstractPortfolio,
        risk_free_rate: int | float | pd.Series = 0,
        window: Optional[int] = None
) -> pd.Series:
    """
    Calculates rolling Sharpe Ratio of a `portfolio`.

    See :func:`~pqr.metrics.sharpe_ratio`.

    Parameters
    ----------
    portfolio
        An allocated portfolio.
    risk_free_rate
        Indicative rate of guaranteed returns (e.g. US government bond rate).
    window
        Number of observations on a rolling window. If not passed, `window`
        equals to approximate number of periods in a year.
    """

    return _roll(
        portfolio.returns,
        metric=_sharpe_ratio,
        window=window,
        risk_free_rate=risk_free_rate
    )


def omega_ratio(
        portfolio: pqr.portfolios.AbstractPortfolio,
        required_return: int | float | pd.Series = 0
) -> int | float:
    """
    Calculates Omega Ratio of a `portfolio`.

    Omega Ratio calculated as the area of the probability distribution function
    of returns above `required_return` divided by the area under
    `required_return`:

    .. math::
        \\Omega(\\theta) = \\frac{\\int_{\\theta}^{\\infty}[1-F(r)]dr}
        {\\int_{-\\infty}^{\\theta}F(r)dr}

    Parameters
    ----------
    portfolio
        An allocated portfolio.
    required_return
        Rate of returns, required for an investor.
    """

    return _omega_ratio(portfolio.returns, required_return)


def rolling_omega_ratio(
        portfolio: pqr.portfolios.AbstractPortfolio,
        required_return: int | float | pd.Series = 0,
        window: Optional[int] = None
) -> pd.Series:
    """
    Calculates rolling Omega Ratio of a `portfolio`.

    See :func:`~pqr.metrics.omega_ratio`.

    Parameters
    ----------
    portfolio
        An allocated portfolio.
    required_return
        Rate of returns, required for an investor.
    window
        Number of observations on a rolling window. If not passed, `window`
        equals to approximate number of periods in a year.
    """

    return _roll(
        portfolio.returns,
        metric=_omega_ratio,
        window=window,
        required_return=required_return
    )


def sortino_ratio(
        portfolio: pqr.portfolios.AbstractPortfolio,
        minimum_acceptable_return: int | float | pd.Series = 0
) -> int | float:
    """
    Calculates Sortino Ratio of a `portfolio`.

    Sortino Ratio is the mean of adjusted by `minimum_acceptable_return`
    `portfolio` returns divided by Downside Risk:

    .. math::
        SR = \\frac{\\bar{r}}{Downside\;Risk}

    Parameters
    ----------
    portfolio
        An allocated portfolio.
    minimum_acceptable_return
        Rate of minimum acceptable returns, required for an investor.
    """

    return _sortino_ratio(portfolio.returns, minimum_acceptable_return)


def rolling_sortino_ratio(
        portfolio: pqr.portfolios.AbstractPortfolio,
        minimum_acceptable_return: int | float | pd.Series = 0,
        window: Optional[int] = None
) -> pd.Series:
    """
    Calculates rolling Sortino Ratio of a `portfolio`.

    See :func:`~pqr.metrics.sortino_ratio`.

    Parameters
    ----------
    portfolio
        An allocated portfolio.
    minimum_acceptable_return
        Rate of returns, required for an investor.
    window
        Number of observations on a rolling window. If not passed, `window`
        equals to approximate number of periods in a year.
    """

    return _roll(
        portfolio.returns,
        metric=_sortino_ratio,
        window=window,
        minimum_acceptable_return=minimum_acceptable_return
    )


def downside_risk(
        portfolio: pqr.portfolios.AbstractPortfolio,
        minimum_acceptable_return: int | float | pd.Series
) -> int | float:
    """
    Calculates Downside Risk of a `portfolio`.

    Downside Risk is the annualized deviation of `portfolio` returns under
    `minimum_acceptable_return`:

    .. math::
        DR = \\sqrt{\\frac{\\sum_{i=1}^{n}max(r_i-mar, 0)}{n}}

    Parameters
    ----------
    portfolio
        An allocated portfolio.
    minimum_acceptable_return
        Rate of minimum acceptable returns, required for an investor.
    """

    return _downside_risk(portfolio.returns, minimum_acceptable_return)


def rolling_downside_risk(
        portfolio: pqr.portfolios.AbstractPortfolio,
        minimum_acceptable_return: int | float | pd.Series = 0,
        window: Optional[int] = None
) -> pd.Series:
    """
    Calculates rolling Downside Risk of a `portfolio`.

    See :func:`~pqr.metrics.downside_risk`.

    Parameters
    ----------
    portfolio
        An allocated portfolio.
    minimum_acceptable_return
        Rate of returns, required for an investor.
    window
        Number of observations on a rolling window. If not passed, `window`
        equals to approximate number of periods in a year.
    """

    return _roll(
        portfolio.returns,
        metric=_downside_risk,
        window=window,
        minimum_acceptable_return=minimum_acceptable_return
    )


def mean_excess_return(
        portfolio: pqr.portfolios.AbstractPortfolio,
        benchmark: pqr.benchmarks.Benchmark
) -> int | float:
    """
    Calculates Mean Excess Return of a `portfolio`.

    Mean Excess Return is the mean difference between `portfolio` returns and
    `benchmark` returns:

    .. math::
        MER = E(r_{portfolio} - r_{benchmark})

    Parameters
    ----------
    portfolio
        An allocated portfolio.
    benchmark
        Benchmark for the `portfolio` to beat.
    """

    return _mean_excess_return(portfolio.returns, benchmark.returns)


def rolling_mean_excess_return(
        portfolio: pqr.portfolios.AbstractPortfolio,
        benchmark: pqr.benchmarks.Benchmark,
        window: Optional[int] = None
) -> pd.Series:
    """
    Calculates rolling Mean Excess Return of a `portfolio`.

    See :func:`~pqr.metrics.mean_excess_return`.

    Parameters
    ----------
    portfolio
        An allocated portfolio.
    benchmark
        Benchmark for the `portfolio` to beat.
    window
        Number of observations on a rolling window. If not passed, `window`
        equals to approximate number of periods in a year.
    """

    return _roll(
        portfolio.returns,
        benchmark.returns,
        metric=_mean_excess_return,
        window=window
    )


def benchmark_correlation(
        portfolio: pqr.portfolios.AbstractPortfolio,
        benchmark: pqr.benchmarks.Benchmark
) -> int | float:
    """
    Calculates Benchmark Correlation of a `portfolio`.

    Benchmark Correlation is the simple spearman correlation between
    `portfolio` returns and `benchmark` returns:

    .. math::
        BC = corr(r_{portfolio}, r_{benchmark})

    Parameters
    ----------
    portfolio
        An allocated portfolio.
    benchmark
        Benchmark for the `portfolio` to beat.
    """

    return _benchmark_correlation(portfolio.returns, benchmark.returns)


def rolling_benchmark_correlation(
        portfolio: pqr.portfolios.AbstractPortfolio,
        benchmark: pqr.benchmarks.Benchmark,
        window: Optional[int] = None
) -> pd.Series:
    """
    Calculates rolling Benchmark Correlation of a `portfolio`.

    See :func:`~pqr.metrics.benchmark_correlation`.

    Parameters
    ----------
    portfolio
        An allocated portfolio.
    benchmark
        Benchmark for the `portfolio` to beat.
    window
        Number of observations on a rolling window. If not passed, `window`
        equals to approximate number of periods in a year.
    """

    return _roll(
        portfolio.returns,
        benchmark.returns,
        metric=_benchmark_correlation,
        window=window
    )


def alpha(
        portfolio: pqr.portfolios.AbstractPortfolio,
        benchmark: pqr.benchmarks.Benchmark,
        risk_free_rate: int | float | pd.Series = 0
) -> int | float:
    """
    Calculates Alpha of a `portfolio`.

    Alpha is the coefficient :math:`\\alpha` in the estimated regression of
    `portfolio` returns per `benchmark` returns (both are adjusted by
    `risk_free_rate`):

    .. math::
        r_{portfolio}-r_f = \\alpha +\\beta*(r_{benchmark}-r_f)+\\epsilon

    Parameters
    ----------
    portfolio
        An allocated portfolio.
    benchmark
        Benchmark for the `portfolio` to beat.
    risk_free_rate
        Indicative rate of guaranteed returns (e.g. US government bond rate).
    """

    return _alpha(portfolio.returns, benchmark.returns, risk_free_rate)


def rolling_alpha(
        portfolio: pqr.portfolios.AbstractPortfolio,
        benchmark: pqr.benchmarks.Benchmark,
        risk_free_rate: int | float | pd.Series = 0,
        window: Optional[int] = None
) -> pd.Series:
    """
    Calculates rolling Alpha of a `portfolio`.

    See :func:`~pqr.metrics.alpha`.

    Parameters
    ----------
    portfolio
        An allocated portfolio.
    benchmark
        Benchmark for the `portfolio` to beat.
    risk_free_rate
        Indicative rate of guaranteed returns (e.g. US government bond rate).
    window
        Number of observations on a rolling window. If not passed, `window`
        equals to approximate number of periods in a year.
    """

    return _roll(
        portfolio.returns,
        benchmark.returns,
        metric=_alpha,
        window=window,
        risk_free_rate=risk_free_rate
    )


def beta(
        portfolio: pqr.portfolios.AbstractPortfolio,
        benchmark: pqr.benchmarks.Benchmark,
        risk_free_rate: int | float | pd.Series = 0
) -> int | float:
    """
    Calculates Beta of a `portfolio`.

    Beta is the coefficient :math:`\\beta` in the estimated regression of
    `portfolio` returns per `benchmark` returns (both are adjusted by
    `risk_free_rate`):

    .. math::
        r_{portfolio}-r_f = \\alpha +\\beta*(r_{benchmark}-r_f)+\\epsilon

    Parameters
    ----------
    portfolio
        An allocated portfolio.
    benchmark
        Benchmark for the `portfolio` to beat.
    risk_free_rate
        Indicative rate of guaranteed returns (e.g. US government bond rate).
    """

    return _beta(portfolio.returns, benchmark.returns, risk_free_rate)


def rolling_beta(
        portfolio: pqr.portfolios.AbstractPortfolio,
        benchmark: pqr.benchmarks.Benchmark,
        risk_free_rate: int | float | pd.Series = 0,
        window: Optional[int] = None
) -> pd.Series:
    """
    Calculates rolling Beta of a `portfolio`.

    See :func:`~pqr.metrics.beta`.

    Parameters
    ----------
    portfolio
        An allocated portfolio.
    benchmark
        Benchmark for the `portfolio` to beat.
    risk_free_rate
        Indicative rate of guaranteed returns (e.g. US government bond rate).
    window
        Number of observations on a rolling window. If not passed, `window`
        equals to approximate number of periods in a year.
    """

    return _roll(
        portfolio.returns,
        benchmark.returns,
        metric=_beta,
        window=window,
        risk_free_rate=risk_free_rate
    )


def _cumulative_returns(returns: pd.Series) -> pd.Series:
    return (1 + returns).cumprod() - 1


def _total_return(returns: pd.Series) -> int | float:
    return _cumulative_returns(returns).iloc[-1]


def _annual_return(returns: pd.Series) -> int | float:
    annualization_factor = _get_annualization_factor(returns)
    years = len(returns) / annualization_factor
    return (1 + _total_return(returns)) ** (1 / years) - 1


def _annual_volatility(returns: pd.Series) -> int | float:
    annualization_rate = np.sqrt(_get_annualization_factor(returns))
    return _volatility(returns) * annualization_rate


def _mean_return(returns: pd.Series) -> int | float:
    return returns.mean()


def _volatility(returns: pd.Series) -> int | float:
    return returns.std(ddof=1)


def _max_drawdown(returns: pd.Series) -> int | float:
    equity_curve = _cumulative_returns(returns) + 1
    high_water_mark = equity_curve.cummax()
    drawdown = (high_water_mark - equity_curve) / high_water_mark
    return -drawdown[np.isfinite(drawdown)].max()


def _win_rate(returns: pd.Series) -> int | float:
    positive_periods = (returns > 0).sum()
    total_periods = len(returns)
    return positive_periods / total_periods


def _value_at_risk(
        returns: pd.Series,
        confidence_level: int | float = 0.95
) -> int | float:
    return returns.quantile(1 - confidence_level)


def _expected_tail_loss(
        returns: pd.Series,
        confidence_level: int | float = 0.95
):
    var = _value_at_risk(returns, 1 - confidence_level)
    return returns[returns <= var].mean()


def _rachev_ratio(
        returns: pd.Series,
        reward_cutoff: int | float = 0.95,
        risk_cutoff: int | float = 0.05,
        risk_free_rate: int | float | pd.Series = 0
) -> int | float:
    adjusted_returns = _adjust_returns(returns, risk_free_rate)
    reward_var = _value_at_risk(adjusted_returns, 1 - reward_cutoff)
    etr = adjusted_returns[adjusted_returns >= reward_var].mean()
    etl = _expected_tail_loss(adjusted_returns, 1 - risk_cutoff)
    return etr / etl


def _calmar_ratio(returns: pd.Series) -> int | float:
    return _annual_return(returns) / _max_drawdown(returns)


def _sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: int | float | pd.Series = 0
) -> int | float:
    adjusted_returns = _adjust_returns(returns, risk_free_rate)
    annualization_rate = np.sqrt(_get_annualization_factor(adjusted_returns))
    return (_mean_return(adjusted_returns) / _volatility(adjusted_returns) *
            annualization_rate)


def _omega_ratio(
        returns: pd.Series,
        required_return: int | float | pd.Series
) -> int | float:
    adjusted_returns = _adjust_returns(returns, required_return)
    above = adjusted_returns[adjusted_returns > 0].sum()
    under = adjusted_returns[adjusted_returns < 0].sum()
    return above / under


def _sortino_ratio(
        returns: pd.Series,
        minimum_acceptable_return: int | float | pd.Series
) -> int | float:
    adjusted_returns = _adjust_returns(returns, minimum_acceptable_return)
    return _mean_return(adjusted_returns) / _downside_risk(adjusted_returns, 0)


def _downside_risk(
        returns: pd.Series,
        minimum_acceptable_return: int | float | pd.Series = 0
) -> int | float:
    adjusted_returns = _adjust_returns(returns, minimum_acceptable_return)
    annualization_rate = np.sqrt(_get_annualization_factor(adjusted_returns))

    returns_under_mar = np.clip(adjusted_returns, a_min=-np.inf, a_max=0)
    downside_risk_ = np.sqrt((returns_under_mar ** 2).mean())
    return downside_risk_ * annualization_rate


def _mean_excess_return(
        returns: pd.Series,
        benchmark_returns: pd.Series
) -> int | float:
    adjusted_returns = _adjust_returns(returns, benchmark_returns)
    return _mean_return(adjusted_returns)


def _benchmark_correlation(
        returns: pd.Series,
        benchmark_returns: pd.Series
) -> int | float:
    trading_available = returns.index.intersection(benchmark_returns.index)
    return returns.corr(benchmark_returns.loc[trading_available])


def _alpha(
        returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: int | float | pd.Series
) -> int | float:
    return _alpha_beta(returns, benchmark_returns, risk_free_rate).iloc[0]


def _beta(
        returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: int | float | pd.Series
) -> int | float:
    return _alpha_beta(returns, benchmark_returns, risk_free_rate).iloc[1]


def _alpha_beta(
        returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: int | float | pd.Series = 0
) -> pd.Series:
    adjusted_returns = _adjust_returns(returns, risk_free_rate)
    adjusted_benchmark_returns = _adjust_returns(benchmark_returns,
                                                 risk_free_rate)
    trading_available = adjusted_returns.index.intersection(
        adjusted_benchmark_returns.index)
    adjusted_benchmark_returns = adjusted_benchmark_returns[trading_available]
    x = sm_tools.add_constant(adjusted_benchmark_returns)
    est = sm_linear.OLS(adjusted_returns, x).fit()
    return est.params


def _adjust_returns(
        returns: pd.Series,
        benchmark_returns: int | float | pd.Series
):
    if isinstance(benchmark_returns, pd.Series):
        available_index = returns.index.intersection(returns.index)
        benchmark_returns = benchmark_returns[available_index]
    return returns - benchmark_returns


def _roll(
        *returns: pd.Series,
        metric: Callable,
        window: Optional[int] = None,
        **kwargs
) -> pd.Series:
    if window is None:
        window = _get_annualization_factor(returns[0])

    common_index = returns[0].index
    for r in returns:
        common_index = common_index.intersection(r.index)

    values = [np.nan] * (window - 1)
    for i in range(window, len(common_index) + 1):
        idx = common_index[i - window:i]
        rets = [r.loc[idx] for r in returns]
        values.append(metric(*rets, **kwargs))
    return pd.Series(values, index=common_index)


def _get_annualization_factor(data: pd.Series | pd.DataFrame) -> int:
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

    inferred_freq = getattr(data.index, 'inferred_freq', None)
    freq_num = freq_alias_to_num.get(inferred_freq)
    if freq_num is None:
        raise ValueError('periodicity of given data cannot be defined, '
                         'try to resample data')
    return freq_num
