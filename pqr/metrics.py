"""This module contains metrics and statistics to assess performance of a portfolio. Usual metrics are
always numbers (int or float), but also rolling metrics are supported. Rolling metrics by default
are calculated to estimate annual performance of a portfolio in each trading period: every period
gather not all points of returns but only for 1 year (e.g. if returns are monthly, rolling window
size equals to 12). But also window size can be given by user.

For now practically all popular in portfolio management metrics and statistics are supported, but
you are welcome to create your own metrics and to contribute to source code.
"""

from collections import namedtuple

import numpy as np
import pandas as pd
from statsmodels.api import OLS, add_constant
from scipy.stats import ttest_1samp

from .utils import get_annualization_factor, align

__all__ = [
    'compound_returns', 'drawdown', 
    'total_return', 'cagr',

    'mean_return', 'rolling_mean_return',
    'volatility', 'rolling_volatility',
    'win_rate', 'rolling_win_rate',
    'max_drawdown', 'rolling_max_drawdown',

    'value_at_risk', 'rolling_value_at_risk',
    'expected_tail_loss', 'rolling_expected_tail_loss',
    'expected_tail_reward', 'rolling_expected_tail_reward',
    'rachev_ratio', 'rolling_rachev_ratio',

    'calmar_ratio', 'rolling_calmar_ratio',
    'sharpe_ratio', 'rolling_sharpe_ratio',
    'omega_ratio', 'rolling_omega_ratio',
    'sortino_ratio', 'rolling_sortino_ratio',

    'mean_excess_return', 'rolling_mean_excess_return',
    'benchmark_correlation', 'rolling_benchmark_correlation',
    'alpha', 'rolling_alpha',
    'beta', 'rolling_beta',

    'turnover',
]


def compound_returns(returns):
    """Calculates Compound Returns of portfolio returns.

    Parameters
    ----------
    returns : pd.Series
        Returns of a portfolio or a benchmark.

    Returns
    -------
    pd.Series
        Compound Returns.

    Notes
    -----
    Starting point equals to zero.
    """

    return (1 + returns).cumprod() - 1


def drawdown(returns):
    """Calculates Drawdown of portfolio returns.

    Drawdown of portfolio is the relative difference between high water mark (cumulative maximum of 
    the compound returns) and compound returns:

    .. math::
        DD = -\\frac{High\\;Water\\;Mark - Compounded\\;Returns}{High\\;Water\\;Mark}

    Parameters
    ----------
    returns : pd.Series
        Returns of a portfolio or a benchmark.

    Returns
    -------
    pd.Series
        Drawdown.
    """

    equity = compound_returns(returns) + 1
    high_water_mark = equity.cummax()
    return -(high_water_mark - equity) / high_water_mark


def total_return(returns):
    """Calculates Total Return of portfolio returns.

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

    return compound_returns(returns).iloc[-1]


def cagr(returns):
    """Calculates Compounded Annual Growth Rate of portfolio returns.

    CAGR calculated as:

    .. math::
        CAGR = (1 + Total Return)^{\\frac{1}{Years}} - 1

    Parameters
    ----------
    returns : pd.Series
        Returns of a portfolio or a benchmark.

    Returns
    -------
    float
        CAGR.
    """

    annualization_factor = get_annualization_factor(returns)
    years = len(returns) / annualization_factor
    return (1 + total_return(returns)) ** (1 / years) - 1


def mean_return(returns):
    """Calculates Mean Return of portfolio returns.

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

    MeanReturn = namedtuple('MeanReturn', ['value', 't_stat', 'p_value'])

    ttest = ttest_1samp(returns, 0, alternative='greater')

    return MeanReturn(value=returns.mean() * get_annualization_factor(returns),
                      t_stat=ttest.statistic, p_value=ttest.pvalue)


def rolling_mean_return(returns, window=None):
    """Calculates rolling Mean Return of portfolio returns.

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

    return _roll(returns, metric=lambda r: mean_return(r).value, window=window)


def volatility(returns):
    """Calculates Volatility of portfolio returns.

    Volatility of the portfolio is the annualized standard deviation of portfolio returns:

    .. math::
        \\sigma_r = \\sqrt{\\frac{\\sum_{i=1}^{n}(r_i-\\bar{r})^2}{n-1}} * 
        \\sqrt{Number\\;of\\;Periods\\;in\\;a\\;Year}

    Parameters
    ----------
    returns : pd.Series
        Returns of a portfolio or a benchmark.

    Returns
    -------
    float
        Volatility.
    """

    annualization_rate = np.sqrt(get_annualization_factor(returns))
    return returns.std(ddof=1) * annualization_rate


def rolling_volatility(returns, window=None):
    """Calculates rolling Volatility of a `portfolio`.

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


def win_rate(returns):
    """Calculates Win Rate of portfolio returns.

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
    """Calculates rolling Win Rate of portfolio returns.

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


def max_drawdown(returns):
    """Calculates Maximum Drawdown of portfolio returns.

    Maximum Drawdown of portfolio is the highest relative difference between high water mark
    (cumulative maximum of the compound returns) and compound returns:

    .. math::
        MDD = \\max\\{Drawdown\\}

    Parameters
    ----------
    returns : pd.Series
        Returns of a portfolio or a benchmark.

    Returns
    -------
    float
        Maximum Drawdown.
    """

    return drawdown(returns).min()


def rolling_max_drawdown(returns, window=None):
    """Calculates rolling Maximum Drawdown of portfolio returns.

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


def value_at_risk(returns, cutoff=0.05):
    """Calculates Value at Risk of portfolio returns.

    VaR shows the amount of potential loss that could happen in a portfolio with given
    `cutoff`:

    .. math::
        VaR = -\\inf\\{F_r(r) > Confidence\\;Level\\}

    Parameters
    ----------
    returns : pd.Series
        Returns of a portfolio or a benchmark.
    cutoff : float, default=0.05
        The probability, with which the estimation of VaR is true.

    Returns
    -------
    float
        Value at Risk.
    """

    return returns.quantile(cutoff) * np.sqrt(get_annualization_factor(returns))


def rolling_value_at_risk(returns, cutoff=0.05, window=None):
    """Calculates rolling Value at Risk of portfolio returns.

    See :func:`~pqr.metrics.value_at_risk`.

    Parameters
    ----------
    returns : pd.Series
        Returns of a portfolio or a benchmark.
    cutoff : float, default=0.05
        The probability, with which the estimation of VaR is true.
    window : int > 0, optional
        Number of observations in a rolling window. If not passed, `window` equals to approximate
        number of periods in a year.

    Returns
    -------
    pd.Series
        Rolling Value at Risk.
    """

    return _roll(returns, metric=value_at_risk, window=window, cutoff=cutoff)


def expected_tail_loss(returns, cutoff=0.05):
    """Calculates Expected Tail Loss of portfolio returns.

    Expected Tail Loss shows the average of the values that fall below the VaR, calculated with
    given `cutoff`:

    .. math::
        ETL = \\frac{\\sum_{i=1}^{n}r_i\\cdot[r_i \\le VaR]}{\\sum_{i=1}^{n}[r_i \\le VaR]}

    Parameters
    ----------
    returns : pd.Series
        Returns of a portfolio or a benchmark.
    cutoff : float, default=0.95
        The probability, with which the estimation of VaR is true.

    Returns
    -------
    float
        Expected Tail Loss.
    """

    var = returns.quantile(cutoff)
    return returns[returns <= var].mean() * np.sqrt(get_annualization_factor(returns))


def rolling_expected_tail_loss(returns, cutoff=0.05, window=None):
    """Calculates rolling Expected Tail Loss of portfolio returns.

    See :func:`~pqr.metrics.expected_tail_loss`.

    Parameters
    ----------
    returns : pd.Series
        Returns of a portfolio or a benchmark.
    cutoff : float, default=0.05
        The probability, with which the estimation of VaR is true.
    window : int > 0, optional
        Number of observations in a rolling window. If not passed, `window` equals to approximate
        number of periods in a year.

    Returns
    -------
    pd.Series
        Rolling Expected Tail Loss.
    """

    return _roll(returns, metric=expected_tail_loss, window=window, cutoff=cutoff)


def expected_tail_reward(returns, cutoff=0.95):
    """Calculates Expected Tail Reward of portfolio returns.

    Expected Tail Loss shows the average of the values that fall beyond the VaR, calculated with
    given `cutoff`:

    .. math::
        ETR = \\frac{\\sum_{i=1}^{n}r_i\\cdot[r_i \\ge VaR]}{\\sum_{i=1}^{n}[r_i \\ge VaR]}

    Parameters
    ----------
    returns : pd.Series
        Returns of a portfolio or a benchmark.
    cutoff : float, default=0.95
        The probability, with which the estimation of VaR is true.

    Returns
    -------
    float
        Expected Tail Reward.
    """

    var = returns.quantile(cutoff)
    return returns[returns >= var].mean() * np.sqrt(get_annualization_factor(returns))


def rolling_expected_tail_reward(returns, cutoff=0.05, window=None):
    """Calculates rolling Expected Tail Reward of portfolio returns.

    See :func:`~pqr.metrics.expected_tail_reward`.

    Parameters
    ----------
    returns : pd.Series
        Returns of a portfolio or a benchmark.
    cutoff : float, default=0.05
        The probability, with which the estimation of VaR is true.
    window : int > 0, optional
        Number of observations in a rolling window. If not passed, `window` equals to approximate
        number of periods in a year.

    Returns
    -------
    pd.Series
        Rolling Expected Tail Reward.
    """

    return _roll(returns, metric=expected_tail_reward, window=window, cutoff=cutoff)


def rachev_ratio(returns, reward_cutoff=0.95, risk_cutoff=0.05, risk_free_rate=0):
    """Calculates Rachev Ratio (R-Ratio) of portfolio returns.

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
        Cutoff to calculate expected tail loss. 
    risk_free_rate : array_like, default=0
        Indicative rate of guaranteed returns (e.g. US government bond rate).

    Returns
    -------
    float
        Rachev Ratio.
    """

    adjusted_returns = _adjust_returns(returns, risk_free_rate)
    etr = expected_tail_reward(adjusted_returns, reward_cutoff)
    etl = expected_tail_loss(adjusted_returns, risk_cutoff)
    return etr / -etl


def rolling_rachev_ratio(returns, reward_cutoff=0.95, risk_cutoff=0.05, risk_free_rate=0,
                         window=None):
    """Calculates rolling Rachev Ratio (R-Ratio) of portfolio returns.

    See :func:`~pqr.metrics.rachev_ratio`.

    Parameters
    ----------
    returns : pd.Series
        Returns of a portfolio or a benchmark.
    reward_cutoff : float, default=0.95
        Cutoff to calculate expected tail return.
    risk_cutoff : float, default=0.05
        Cutoff to calculate expected tail loss.
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
    """Calculates Calmar Ratio of portfolio returns.

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

    return cagr(returns) / -max_drawdown(returns)


def rolling_calmar_ratio(returns, window=None):
    """Calculates rolling Calmar Ratio of portfolio returns.

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
    """Calculates Sharpe Ratio of portfolio returns.

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
    return mean_return(adjusted_returns).value / volatility(adjusted_returns)


def rolling_sharpe_ratio(returns, risk_free_rate=0, window=None) -> pd.Series:
    """Calculates rolling Sharpe Ratio of portfolio returns.

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
    """Calculates Omega Ratio of portfolio returns.

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
    """Calculates rolling Omega Ratio of a portfolio returns.

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
    """Calculates Sortino Ratio of portfolio returns.

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
    downside_risk = np.sqrt((returns_under_mar ** 2).mean())

    return adjusted_returns.mean() / downside_risk * annualization_rate


def rolling_sortino_ratio(returns, minimum_acceptable_return=0, window=None):
    """Calculates rolling Sortino Ratio of portfolio returns.

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


def benchmark_correlation(returns, benchmark):
    """Calculates Benchmark Correlation of portfolio returns.

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
    """Calculates rolling Benchmark Correlation of portfolio returns.

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


def mean_excess_return(returns, benchmark):
    """Calculates Mean Excess Return of portfolio returns.

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

    MeanExcessReturn = namedtuple('MeanExcessReturn', ['value', 't_stat', 'p_value'])

    excess_returns = _adjust_returns(returns, benchmark)
    ttest = ttest_1samp(excess_returns, 0, alternative='greater')
    return MeanExcessReturn(value=excess_returns.mean() * get_annualization_factor(returns),
                            t_stat=ttest.statistic, p_value=ttest.pvalue)


def rolling_mean_excess_return(returns, benchmark, window=None) -> pd.Series:
    """Calculates rolling Mean Excess Return of portfolio returns.

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

    return _roll(returns, benchmark, metric=lambda r, b: mean_excess_return(r, b).value, window=window)


def alpha(returns, benchmark, risk_free_rate=0):
    """Calculates Annualized Alpha of portfolio returns.

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

    Alpha = namedtuple('Alpha', ['value', 't_stat', 'p_value'])
    ols_est = _alpha_beta(returns, benchmark, risk_free_rate)
    return Alpha(value=ols_est.params[0] * get_annualization_factor(returns),
                 t_stat=ols_est.tvalues[0], p_value=ols_est.pvalues[0])


def rolling_alpha(returns, benchmark, risk_free_rate=0, window=None):
    """Calculates rolling Alpha of portfolio returns.

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
    benchmark = _adjust_returns(benchmark, risk_free_rate)
    return _roll(returns, benchmark, 
                 metric=lambda r, b, **kw: alpha(r, b, **kw).value, 
                 window=window, risk_free_rate=0)


def beta(returns, benchmark, risk_free_rate=0):
    """Calculates Beta of portfolio returns.

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

    Beta = namedtuple('Beta', ['value', 't_stat', 'p_value'])
    ols_est = _alpha_beta(returns, benchmark, risk_free_rate)
    return Beta(value=ols_est.params[1],
                t_stat=ols_est.tvalues[1], p_value=ols_est.pvalues[1])


def rolling_beta(returns, benchmark, risk_free_rate=0, window=None) -> pd.Series:
    """Calculates rolling Beta of portfolio returns.

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
    return _roll(returns, benchmark, 
                 metric=lambda r, b, **kw: beta(r, b, **kw).value, 
                 window=window, risk_free_rate=0)

def turnover(positions):
    long, short = positions > 0, positions < 0
    with np.errstate(divide='ignore', invalid='ignore'):
        positions_long = np.where(long, positions, 0).astype(float)
        positions_long /= np.nansum(positions_long, axis=1, keepdims=True).astype(float)
        positions_long = pd.DataFrame(positions_long)

        positions_short = np.where(short, positions, 0).astype(float)
        positions_short /= np.nansum(positions_short, axis=1, keepdims=True).astype(float)
        positions_short = pd.DataFrame(positions_short)

    turnover_long = np.nansum(positions_long.diff().abs(), axis=0)
    turnover_short = np.nansum(positions_short.diff().abs(), axis=0)
    periodical_turnover = np.nansum(turnover_long + turnover_short)

    return periodical_turnover.mean()


def _adjust_returns(returns, adjustment):
    if isinstance(adjustment, pd.Series):
        returns, adjustment = returns.align(adjustment, join='inner')
    return returns - adjustment


def _alpha_beta(returns, benchmark, risk_free_rate=0):
    adjusted_returns = _adjust_returns(returns, risk_free_rate)
    adjusted_benchmark = _adjust_returns(benchmark, risk_free_rate)
    adjusted_returns, adjusted_benchmark = adjusted_returns.align(adjusted_benchmark, join='inner')
    x = add_constant(adjusted_benchmark.values)
    est = OLS(adjusted_returns.values, x).fit()
    return est


def _roll(*returns, metric, window=None, **kwargs):
    if window is None:
        window = get_annualization_factor(returns[0])

    returns = align(*returns)
    index = returns[0].index

    values = []
    for i in range(window, len(index) + 1):
        idx = index[i - window:i]
        rets = [r.loc[idx] for r in returns]
        values.append(metric(*rets, **kwargs))
    return pd.Series(values, index=index[window-1:])
