"""
This module includes instruments to analyze, what are the drivers of returns of a portfolio. It can
be used to identify, whether excess returns are achieved by skills of portfolio managing or simply
by exposure on some factors.
"""

import pandas as pd
import statsmodels.regression.linear_model as sm_linear
import statsmodels.tools.tools as sm_tools

from .utils import get_annualization_factor, align


def explain_alpha(returns, market_returns, factors_returns, risk_free_rate=0):
    """
    Calculates alpha and betas on factors, including overall market.

    Parameters
    ----------
    returns : pd.Series
        Returns of a portfolio.
    market_returns : pd.Series
        Returns of overall market or some other benchmark for the portoflio.
    factors_returns : sequence of pd.Series
        Returns of other factors.
    risk_free_rate : array_like, default=0
        Indicative rate of guaranteed returns (e.g. US government bond rate).

    Returns
    -------
    pd.DataFrame
        Table with alpha and betas on factors with values, showing their statistical significance.
    """

    returns, market_returns, *factors_returns = align(returns, market_returns, *factors_returns)

    adjusted_returns = returns - risk_free_rate
    adjusted_market_returns = market_returns - risk_free_rate

    factors_returns = pd.DataFrame([adjusted_market_returns] + factors_returns).T
    factors_returns = sm_tools.add_constant(factors_returns)

    y, x = align(adjusted_returns, factors_returns)
    est = sm_linear.OLS(y, x).fit()
    params = est.params.values
    params[0] *= get_annualization_factor(returns)

    return pd.DataFrame(
        [params, est.tvalues.values, est.pvalues.values],
        index=['value', 't-stat', 'p-value'],
        columns=['alpha'] + [f'beta_{factor}' for factor in factors_returns.iloc[:, 1:]]
    )
