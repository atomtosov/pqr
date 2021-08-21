"""
This module contains metrics and statistics to assess performance of a portfolio. Usual metrics are
always numbers (int or float), but also rolling metrics are supported. Rolling metrics by default
are calculated to estimate annual performance of a portfolio in each trading period: every period
gather not all points of returns but only for 1 year (e.g. if returns are monthly, rolling window
size equals to 12). But also window size can be given by user.

For now practically all popular in portfolio management metrics and statistics are supported, but
you are welcome to create your own metrics and to contribute to source code.
"""

import numpy as np
import pandas as pd
import statsmodels.regression.linear_model as sm_linear
import statsmodels.tools.tools as sm_tools

from .utils import get_annualization_factor, align

__all__ = [
    'summary',

    'cumulative_returns', 'total_return',
    'annual_return', 'annual_volatility',

    'mean_return', 'rolling_mean_return',
    'win_rate', 'rolling_win_rate',
    'volatility', 'rolling_volatility',
    'max_drawdown', 'rolling_max_drawdown',

    'value_at_risk', 'rolling_value_at_risk',
    'expected_tail_loss', 'rolling_expected_tail_loss',
    'rachev_ratio', 'rolling_rachev_ratio',

    'calmar_ratio', 'rolling_calmar_ratio',
    'sharpe_ratio', 'rolling_sharpe_ratio',
    'omega_ratio', 'rolling_omega_ratio',
    'sortino_ratio', 'rolling_sortino_ratio',

    'mean_excess_return', 'rolling_mean_excess_return',
    'benchmark_correlation', 'rolling_benchmark_correlation',
    'alpha', 'rolling_alpha',
    'beta', 'rolling_beta',
]


def summary(portfolio, benchmark):
    """
    Calculates summary statistics for a `portfolio`.

    Computed metrics:

    * Total Return, %
    * Annual Return, %
    * Annual Volatility, %
    * Mean Return, %
    * Win Rate, %
    * Volatility, %
    * Maximum Drawdown, %
    * VaR, %
    * Expected Tail Loss, %
    * Rachev Ratio
    * Calmar Ratio
    * Omega Ratio
    * Sortino Ratio
    * Mean Excess Return, %
    * Benchmark Correlation
    * Alpha, %
    * Beta

    Parameters
    ----------
    portfolio : Portfolio
        Portfolio, for which metrics are calculated.
    benchmark : Benchmark or Portfolio
        Benchmark, which used as the alternative for the `portfolio` to calculate some metrics.

    Returns
    -------
    pd.Series
    """

    return pd.Series(
        {
            'Total Return, %': total_return(portfolio.returns) * 100,
            'Annual Return, %': annual_return(portfolio.returns) * 100,
            'Annual Volatility, %': annual_volatility(portfolio.returns) * 100,
            'Mean Return, %': mean_return(portfolio.returns) * 100,
            'Win Rate, %': win_rate(portfolio.returns) * 100,
            'Volatility, %': volatility(portfolio.returns) * 100,
            'Maximum Drawdown, %': max_drawdown(portfolio.returns) * 100,
            'VaR, %': value_at_risk(portfolio.returns) * 100,
            'Expected Tail Loss, %': expected_tail_loss(portfolio.returns) * 100,
            'Rachev Ratio': rachev_ratio(portfolio.returns),
            'Calmar Ratio': calmar_ratio(portfolio.returns),
            'Sharpe Ratio': sharpe_ratio(portfolio.returns),
            'Omega Ratio': omega_ratio(portfolio.returns),
            'Sortino Ratio': sortino_ratio(portfolio.returns),
            'Mean Excess Return, %': mean_excess_return(portfolio.returns, benchmark.returns) * 100,
            'Benchmark Correlation': benchmark_correlation(portfolio.returns, benchmark.returns),
            'Alpha, %': alpha(portfolio.returns, benchmark.returns) * 100,
            'Beta': beta(portfolio.returns, benchmark.returns),
        },
        name=portfolio.name
    ).round(2)


def cumulative_returns(returns):
    """
    Calculates Cumulative Returns of portfolio returns.

    Parameters
    ----------
    returns : pd.Series
        Returns of a portfolio or a benchmark.

    Returns
    -------
    pd.Series
        Cumulative Returns.

    Notes
    -----
    Starting point equals to zero.
    """

    return (1 + returns).cumprod() - 1


def total_return(returns):
    """
    Calculates Total Return of portfolio returns.

    Parameters
    ----------
    returns : pd.Series
        Returns of a portfolio or a benchmark.

    Returns
    -------
    float
        Total Return.

    Notes
    -----
    Shows additional return, not equity curve's state.
    """

    return cumulative_returns(returns).iloc[-1]


def annual_return(returns):
    """
    Calculates Annual Return of portfolio returns as CAGR.

    CAGR (Compounded Annual Growth Rate) calculated as:

    .. math::
        CAGR = (1 + Total Return)^{\\frac{1}{Years}} - 1

    Parameters
    ----------
    returns : pd.Series
        Returns of a portfolio or a benchmark.

    Returns
    -------
    float
        Annual Return.
    """

    annualization_factor = get_annualization_factor(returns)
    years = len(returns) / annualization_factor
    return (1 + total_return(returns)) ** (1 / years) - 1


def annual_volatility(returns):
    """
    Calculates Annual Volatility of portfolio returns.

    Annual Volatility of the portfolio is the annualized standard deviation of portfolio returns:

    .. math::
        \\sigma_r * \\sqrt{Number\;of\;Periods\;in\;a\;Year}

    Parameters
    ----------
    returns : pd.Series
        Returns of a portfolio or a benchmark.

    Returns
    -------
    float
        Annual Volatility.
    """

    annualization_rate = np.sqrt(get_annualization_factor(returns))
    return volatility(returns) * annualization_rate


def mean_return(returns):
    """
    Calculates Mean Return of portfolio returns.

    Mean Return is simple expected value:

    .. math::
        E(r) = \\bar{r}

    Parameters
    ----------
    returns : pd.Series
        Returns of a portfolio or a benchmark.

    Returns
    -------
    float
        Mean Return.
    """

    return returns.mean()


def rolling_mean_return(returns, window=None):
    """
    Calculates rolling Mean Return of portfolio returns.

    See :func:`~pqr.metrics.mean_return`.

    Parameters
    ----------
    returns : pd.Series
        Returns of a portfolio or a benchmark.
    window : int > 0, optional
        Number of observations in a rolling window. If not passed, `window` equals to approximate
        number of periods in a year.

    Returns
    -------
    pd.Series
        Rolling Mean Return.
    """

    return _roll(returns, metric=mean_return, window=window)


def win_rate(returns):
    """
    Calculates Win Rate of portfolio returns.

    Win Rate of a portfolio is simple ratio of number of periods with positive returns and total
    number of trading periods:

    .. math::
        WR = \\frac{\\sum_{i=1}^{n}[r_{i} > 0]}{n}

    Parameters
    ----------
    returns : pd.Series
        Returns of a portfolio or a benchmark.

    Returns
    -------
    float
        Win Rate.
    """

    positive_periods = (returns > 0).sum()
    total_periods = len(returns)
    return positive_periods / total_periods


def rolling_win_rate(returns, window=None):
    """
    Calculates rolling Win Rate of portfolio returns.

    See :func:`~pqr.metrics.win_rate`.

    Parameters
    ----------
    returns : pd.Series
        Returns of a portfolio or a benchmark.
    window : int > 0, optional
        Number of observations in a rolling window. If not passed, `window` equals to approximate
        number of periods in a year.

    Returns
    -------
    pd.Series
        Rolling Win Rate.
    """

    return _roll(returns, metric=win_rate, window=window)


def volatility(returns):
    """
    Calculates Volatility of portfolio returns.

    Volatility of the portfolio returns is the standard deviation:

    .. math::
        \\sigma_r = \\sqrt{\\frac{\\sum_{i=1}^{n}(r_i-\\bar{r})^2}{n-1}}

    Parameters
    ----------
    returns : pd.Series
        Returns of a portfolio or a benchmark.

    Returns
    -------
    float
        Volatility.
    """

    return returns.std(ddof=1)


def rolling_volatility(returns, window=None):
    """
    Calculates rolling Volatility of a `portfolio`.

    See :func:`~pqr.metrics.volatility`.

    Parameters
    ----------
    returns : pd.Series
        Returns of a portfolio or a benchmark.
    window : int > 0, optional
        Number of observations in a rolling window. If not passed, `window` equals to approximate
        number of periods in a year.

    Returns
    -------
    pd.Series
        Rolling Volatility.
    """

    return _roll(returns, metric=volatility, window=window)


def max_drawdown(returns):
    """
    Calculates Maximum Drawdown of portfolio returns.

    Maximum Drawdown of portfolio is the highest relative difference between high water mark
    (cumulative maximum of the cumulative returns) and cumulative returns:

    .. math::
        MDD = -\max{\\frac{HWM - Cumulative\;Returns}{HWM}}

    Parameters
    ----------
    returns : pd.Series
        Returns of a portfolio or a benchmark.

    Returns
    -------
    float
        Maximum Drawdown.
    """

    cumsum_returns = returns.cumsum()
    underwater = cumsum_returns - cumsum_returns.cummax()
    return underwater.min()


def rolling_max_drawdown(returns, window=None):
    """
    Calculates rolling Maximum Drawdown of portfolio returns.

    See :func:`~pqr.metrics.max_drawdown`.

    Parameters
    ----------
    returns : pd.Series
        Returns of a portfolio or a benchmark.
    window : int > 0, optional
        Number of observations in a rolling window. If not passed, `window` equals to approximate
        number of periods in a year.

    Returns
    -------
    pd.Series
        Rolling Maximum Drawdown.
    """

    return _roll(returns, metric=max_drawdown, window=window)


def value_at_risk(returns, confidence_level=0.95):
    """
    Calculates Value at Risk of portfolio returns.

    VaR shows the amount of potential loss that could happen in a portfolio with given
    `confidence_level`:

    .. math::
        VaR = -\\inf\{F_r(r) > Confidence\;Level\}

    Parameters
    ----------
    returns : pd.Series
        Returns of a portfolio or a benchmark.
    confidence_level : float, default=0.95
        The probability, with which the estimation of VaR is true.

    Returns
    -------
    float
        Value at Risk.
    """

    return returns.quantile(1 - confidence_level)


def rolling_value_at_risk(returns, confidence_level=0.95, window=None):
    """
    Calculates rolling Value at Risk of portfolio returns.

    See :func:`~pqr.metrics.value_at_risk`.

    Parameters
    ----------
    returns : pd.Series
        Returns of a portfolio or a benchmark.
    confidence_level : float, default=0.95
        The probability, with which the estimation of VaR is true.
    window : int > 0, optional
        Number of observations in a rolling window. If not passed, `window` equals to approximate
        number of periods in a year.

    Returns
    -------
    pd.Series
        Rolling Value at Risk.
    """

    return _roll(returns, metric=value_at_risk, window=window, confidence_level=confidence_level)


def expected_tail_loss(returns, confidence_level=0.95):
    """
    Calculates Expected Tail Loss of portfolio returns.

    Expected Tail Loss shows the average of the values that fall beyond the VaR, calculated with
    given `confidence_level`:

    .. math::
        ETL = \\frac{\\sum_{i=1}^{n}r_i\cdot[r_i \le VaR]}
        {\\sum_{i=1}^{n}[r_i \le VaR]}

    Parameters
    ----------
    returns : pd.Series
        Returns of a portfolio or a benchmark.
    confidence_level : float, default=0.95
        The probability, with which the estimation of VaR is true.

    Returns
    -------
    float
        Expected Tail Loss.
    """

    var = value_at_risk(returns, confidence_level)
    return returns[returns <= var].mean()


def rolling_expected_tail_loss(returns, confidence_level=0.95, window=None):
    """
    Calculates rolling Expected Tail Loss of portfolio returns.

    See :func:`~pqr.metrics.expected_tail_loss`.

    Parameters
    ----------
    returns : pd.Series
        Returns of a portfolio or a benchmark.
    confidence_level : float, default=0.95
        The probability, with which the estimation of VaR is true.
    window : int > 0, optional
        Number of observations in a rolling window. If not passed, `window` equals to approximate
        number of periods in a year.

    Returns
    -------
    pd.Series
        Rolling Expected Tail Loss.
    """

    return _roll(returns, metric=expected_tail_loss, window=window,
                 confidence_level=confidence_level)


def rachev_ratio(returns, reward_cutoff=0.95, risk_cutoff=0.05, risk_free_rate=0):
    """
    Calculates Rachev Ratio (R-Ratio) of portfolio returns.

    Rachev Ratio calculated as ETR (Expected Tail Return) divided by ETL (Expected Tail Return) of
    adjusted by `risk_free_rate` returns:

    .. math::
        R-Ratio = \\frac{ETR(r)_\\alpha}{ETL(r)_\\beta}

    Parameters
    ----------
    returns : pd.Series
        Returns of a portfolio or a benchmark.
    reward_cutoff : float, default=0.95
        Cutoff to calculate expected tail return.
    risk_cutoff : float, default=0.05
        Cutoff to calculate expected tail loss. Confidence level to compute it
        equals to 1 - `risk_cutoff`.
    risk_free_rate : array_like, default=0
        Indicative rate of guaranteed returns (e.g. US government bond rate).

    Returns
    -------
    float
        Rachev Ratio.
    """

    adjusted_returns = _adjust_returns(returns, risk_free_rate)
    reward_var = value_at_risk(adjusted_returns, 1 - reward_cutoff)
    etr = adjusted_returns[adjusted_returns >= reward_var].mean()
    etl = expected_tail_loss(adjusted_returns, 1 - risk_cutoff)
    return etr / -etl


def rolling_rachev_ratio(returns, reward_cutoff=0.95, risk_cutoff=0.05, risk_free_rate=0,
                         window=None):
    """
    Calculates rolling Rachev Ratio (R-Ratio) of portfolio returns.

    See :func:`~pqr.metrics.rachev_ratio`.

    Parameters
    ----------
    returns : pd.Series
        Returns of a portfolio or a benchmark.
    reward_cutoff : float, default=0.95
        Cutoff to calculate expected tail return.
    risk_cutoff : float, default=0.05
        Cutoff to calculate expected tail loss. Confidence level to compute it
        equals to 1 - `risk_cutoff`.
    risk_free_rate : array_like, default=0
        Indicative rate of guaranteed returns (e.g. US government bond rate).
    window : int > 0, optional
        Number of observations in a rolling window. If not passed, `window` equals to approximate
        number of periods in a year.

    Returns
    -------
    pd.Series
        Rolling Rachev Ratio.
    """

    return _roll(returns, metric=rachev_ratio, window=window, reward_cutoff=reward_cutoff,
                 risk_cutoff=risk_cutoff, risk_free_rate=risk_free_rate)


def calmar_ratio(returns):
    """
    Calculates Calmar Ratio of portfolio returns.

    Calmar Ratio is annual return (CAGR) divided by maximum drawdown of the period:

    .. math::
        CR = \\frac{CAGR(r)}{MDD(r)}

    Parameters
    ----------
    returns : pd.Series
        Returns of a portfolio or a benchmark.

    Returns
    -------
    float
        Calmar Ratio.
    """

    return annual_return(returns) / -max_drawdown(returns)


def rolling_calmar_ratio(returns, window=None):
    """
    Calculates rolling Calmar Ratio of portfolio returns.

    See :func:`~pqr.metrics.calmar_ratio`.

    Parameters
    ----------
    returns : pd.Series
        Returns of a portfolio or a benchmark.
    window : int > 0, optional
        Number of observations in a rolling window. If not passed, `window` equals to approximate
        number of periods in a year.

    Returns
    -------
    pd.Series
        Rolling Calmar Ratio.
    """

    return _roll(returns, metric=calmar_ratio, window=window)


def sharpe_ratio(returns, risk_free_rate=0):
    """
    Calculates Sharpe Ratio of portfolio returns.

    Sharpe Ratio calculated as annualized ratio between mean and volatility of adjusted by
    `risk_free_rate` returns:

    .. math::
        SR = \\frac{\\bar{r}}{\\sigma_{r}}

    Parameters
    ----------
    returns : pd.Series
        Returns of a portfolio or a benchmark.
    risk_free_rate : array_like, default=0
        Indicative rate of guaranteed returns (e.g. US government bond rate).

    Returns
    -------
    float
        Sharpe Ratio.
    """

    adjusted_returns = _adjust_returns(returns, risk_free_rate)
    annualization_rate = np.sqrt(get_annualization_factor(adjusted_returns))
    return mean_return(adjusted_returns) / volatility(adjusted_returns) * annualization_rate


def rolling_sharpe_ratio(returns, risk_free_rate=0, window=None) -> pd.Series:
    """
    Calculates rolling Sharpe Ratio of portfolio returns.

    See :func:`~pqr.metrics.sharpe_ratio`.

    Parameters
    ----------
    returns : pd.Series
        Returns of a portfolio or a benchmark.
    risk_free_rate : array_like, default=0
        Indicative rate of guaranteed returns (e.g. US government bond rate).
    window : int > 0, optional
        Number of observations in a rolling window. If not passed, `window` equals to approximate
        number of periods in a year.

    Returns
    -------
    pd.Series
        Rolling Sharpe Ratio.
    """

    returns = _adjust_returns(returns, risk_free_rate)
    return _roll(returns, metric=sharpe_ratio, window=window, risk_free_rate=0)


def omega_ratio(returns, required_return=0):
    """
    Calculates Omega Ratio of portfolio returns.

    Omega Ratio calculated as the area of the probability distribution function of returns above
    `required_return` divided by the area under `required_return`:

    .. math::
        \\Omega(\\theta) = \\frac{\\int_{\\theta}^{\\infty}[1-F(r)]dr}
        {\\int_{-\\infty}^{\\theta}F(r)dr}

    Parameters
    ----------
    returns : pd.Series
        Returns of a portfolio or a benchmark.
    required_return : array_like, default=0
        Rate of returns, required for an investor.

    Returns
    -------
    float
        Omega Ratio.
    """

    adjusted_returns = _adjust_returns(returns, required_return)
    above = adjusted_returns[adjusted_returns > 0].sum()
    under = -adjusted_returns[adjusted_returns < 0].sum()
    return above / under


def rolling_omega_ratio(returns, required_return=0, window=None) -> pd.Series:
    """
    Calculates rolling Omega Ratio of a portfolio returns.

    See :func:`~pqr.metrics.omega_ratio`.

    Parameters
    ----------
    returns : pd.Series
        Returns of a portfolio or a benchmark.
    required_return : array_like, default=0
        Rate of returns, required for an investor.
    window : int > 0, optional
        Number of observations in a rolling window. If not passed, `window` equals to approximate
        number of periods in a year.

    Returns
    -------
    pd.Series
        Rolling Omega Ratio.
    """

    returns = _adjust_returns(returns, required_return)
    return _roll(returns, metric=omega_ratio, window=window, required_return=0)


def sortino_ratio(returns, minimum_acceptable_return=0):
    """
    Calculates Sortino Ratio of portfolio returns.

    Sortino Ratio is the mean of adjusted by `minimum_acceptable_return` (mar) `portfolio` returns
    divided by Downside Risk:

    .. math::
        SR = \\frac{\\overline{r - mar}}{\\sqrt{\\frac{\\sum_{i=1}^{n}max(r_i-mar, 0)}{n}}}

    Parameters
    ----------
    returns : pd.Series
        Returns of a portfolio or a benchmark.
    minimum_acceptable_return : array_like, default=0
        Rate of minimum acceptable returns, required for an investor.

    Returns
    -------
    float
        Sortino Ratio.
    """

    adjusted_returns = _adjust_returns(returns, minimum_acceptable_return)
    annualization_rate = np.sqrt(get_annualization_factor(adjusted_returns))

    returns_under_mar = np.clip(adjusted_returns, a_min=-np.inf, a_max=0)
    downside_risk_ = np.sqrt((returns_under_mar ** 2).mean())

    return mean_return(adjusted_returns) / downside_risk_ * annualization_rate


def rolling_sortino_ratio(returns, minimum_acceptable_return=0, window=None):
    """
    Calculates rolling Sortino Ratio of portfolio returns.

    See :func:`~pqr.metrics.sortino_ratio`.

    Parameters
    ----------
    returns : pd.Series
        Returns of a portfolio or a benchmark.
    minimum_acceptable_return : array_like, default=0
        Rate of minimum acceptable returns, required for an investor.
    window : int > 0, optional
        Number of observations in a rolling window. If not passed, `window` equals to approximate
        number of periods in a year.

    Returns
    -------
    pd.Series
        Rolling Sortino Ratio.
    """

    returns = _adjust_returns(returns, minimum_acceptable_return)
    return _roll(returns, metric=sortino_ratio, window=window, minimum_acceptable_return=0)


def mean_excess_return(returns, benchmark):
    """
    Calculates Mean Excess Return of portfolio returns.

    Mean Excess Return is the mean difference between portfolio `returns` and `benchmark` returns:

    .. math::
        MER = E(r_{portfolio} - r_{benchmark})

    Parameters
    ----------
    returns : pd.Series
        Returns of a portfolio or a benchmark.
    benchmark : pd.Series
        Benchmark for the portfolio to beat.

    Returns
    -------
    float
        Mean Excess Return.
    """

    adjusted_returns = _adjust_returns(returns, benchmark)
    return mean_return(adjusted_returns)


def rolling_mean_excess_return(returns, benchmark, window=None) -> pd.Series:
    """
    Calculates rolling Mean Excess Return of portfolio returns.

    See :func:`~pqr.metrics.mean_excess_return`.

    Parameters
    ----------
    returns : pd.Series
        Returns of a portfolio or a benchmark.
    benchmark : pd.Series
        Benchmark for the portfolio to beat.
    window : int > 0, optional
        Number of observations in a rolling window. If not passed, `window` equals to approximate
        number of periods in a year.

    Returns
    -------
    pd.Series
        Rolling Mean Excess Return.
    """

    return _roll(returns, benchmark, metric=mean_excess_return, window=window)


def benchmark_correlation(returns, benchmark):
    """
    Calculates Benchmark Correlation of portfolio returns.

    Benchmark Correlation is the simple spearman correlation between portfolio `returns` and
    `benchmark` returns:

    .. math::
        BC = corr(r_{portfolio}, r_{benchmark})

    Parameters
    ----------
    returns : pd.Series
        Returns of a portfolio or a benchmark.
    benchmark : pd.Series
        Benchmark for the portfolio to beat.

    Returns
    -------
    float
        Benchmark Correlation.
    """

    trading_available = returns.index.intersection(benchmark.index)
    return returns.corr(benchmark.loc[trading_available])


def rolling_benchmark_correlation(returns, benchmark, window=None):
    """
    Calculates rolling Benchmark Correlation of portfolio returns.

    See :func:`~pqr.metrics.benchmark_correlation`.

    Parameters
    ----------
    returns : pd.Series
        Returns of a portfolio or a benchmark.
    benchmark : pd.Series
        Benchmark for the portfolio to beat.
    window : int > 0, optional
        Number of observations in a rolling window. If not passed, `window` equals to approximate
        number of periods in a year.

    Returns
    -------
    pd.Series
        Rolling Benchmark Correlation.
    """

    return _roll(returns, benchmark, metric=benchmark_correlation, window=window)


def alpha(returns, benchmark, risk_free_rate=0):
    """
    Calculates Alpha of portfolio returns.

    Alpha is the coefficient :math:`\\alpha` in the estimated regression of portfolio `returns` per
    `benchmark` returns (both are adjusted by `risk_free_rate`):

    .. math::
        r_{portfolio}-r_f = \\alpha +\\beta*(r_{benchmark}-r_f)+\\epsilon

    Parameters
    ----------
    returns : pd.Series
        Returns of a portfolio or a benchmark.
    benchmark : pd.Series
        Benchmark for the portfolio to beat.
    risk_free_rate : array_like, default=0
        Indicative rate of guaranteed returns (e.g. US government bond rate).

    Returns
    -------
    float
        Alpha.
    """

    return _alpha_beta(returns, benchmark, risk_free_rate).iloc[0]


def rolling_alpha(returns, benchmark, risk_free_rate=0, window=None):
    """
    Calculates rolling Alpha of portfolio returns.

    See :func:`~pqr.metrics.alpha`.

    Parameters
    ----------
    returns : pd.Series
        Returns of a portfolio or a benchmark.
    benchmark : pd.Series
        Benchmark for the portfolio to beat.
    risk_free_rate : array_like, default=0
        Indicative rate of guaranteed returns (e.g. US government bond rate).
    window : int > 0, optional
        Number of observations in a rolling window. If not passed, `window` equals to approximate
        number of periods in a year.

    Returns
    -------
    pd.Series
        Rolling Alpha.
    """

    returns = _adjust_returns(returns, risk_free_rate)
    benchmark = _adjust_returns(returns, risk_free_rate)
    return _roll(returns, benchmark, metric=alpha, window=window, risk_free_rate=0)


def beta(returns, benchmark, risk_free_rate=0):
    """
    Calculates Beta of portfolio returns.

    Beta is the coefficient :math:`\\beta` in the estimated regression of portfolio `returns` per
    `benchmark` returns (both are adjusted by `risk_free_rate`):

    .. math::
        r_{portfolio}-r_f = \\alpha +\\beta*(r_{benchmark}-r_f)+\\epsilon

    Parameters
    ----------
    returns : pd.Series
        Returns of a portfolio or a benchmark.
    benchmark : pd.Series
        Benchmark for the portfolio to beat.
    risk_free_rate : array_like, default=0
        Indicative rate of guaranteed returns (e.g. US government bond rate).

    Returns
    -------
    float
        Beta.
    """

    return _alpha_beta(returns, benchmark, risk_free_rate).iloc[1]


def rolling_beta(returns, benchmark, risk_free_rate=0, window=None) -> pd.Series:
    """
    Calculates rolling Beta of portfolio returns.

    See :func:`~pqr.metrics.beta`.

    Parameters
    ----------
    returns : pd.Series
        Returns of a portfolio or a benchmark.
    benchmark : pd.Series
        Benchmark for the portfolio to beat.
    risk_free_rate : array_like, default=0
        Indicative rate of guaranteed returns (e.g. US government bond rate).
    window : int > 0, optional
        Number of observations in a rolling window. If not passed, `window` equals to approximate
        number of periods in a year.

    Returns
    -------
    pd.Series
        Rolling Beta
    """

    returns = _adjust_returns(returns, risk_free_rate)
    benchmark = _adjust_returns(benchmark, risk_free_rate)
    return _roll(returns, benchmark, metric=beta, window=window, risk_free_rate=0)


def _adjust_returns(returns, adjustment):
    if isinstance(adjustment, pd.Series):
        returns, adjustment = align(returns, adjustment)
    return returns - adjustment


def _alpha_beta(returns, benchmark, risk_free_rate=0):
    adjusted_returns = _adjust_returns(returns, risk_free_rate)
    adjusted_benchmark = _adjust_returns(benchmark, risk_free_rate)
    adjusted_returns, adjusted_benchmark = align(adjusted_returns, adjusted_benchmark)
    x = sm_tools.add_constant(adjusted_benchmark)
    est = sm_linear.OLS(adjusted_returns, x).fit()
    return est.params


def _roll(*returns, metric, window=None, **kwargs):
    if window is None:
        window = get_annualization_factor(returns[0])

    common_index = returns[0].index
    for r in returns:
        common_index = common_index.intersection(r.index)

    values = [np.nan] * (window - 1)
    for i in range(window, len(common_index) + 1):
        idx = common_index[i - window:i]
        rets = [r.loc[idx] for r in returns]
        values.append(metric(*rets, **kwargs))
    return pd.Series(values, index=common_index)
