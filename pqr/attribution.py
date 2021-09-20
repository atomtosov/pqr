"""This module includes tools to analyze, what are the drivers of returns of a portfolio. It can
be used to identify, whether excess returns are achieved by skills of portfolio managing or simply
by exposure on some factors. This also can help to more accurately understand the startegy itself and
its differences from simple investing into a factor.
"""

import pandas as pd
from statsmodels.api import OLS, add_constant

from .utils import align


def explain_alpha(returns, market_returns, factors_returns, risk_free_rate=0):
    """Calculates alpha and betas on factors, including overall market.

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
    factors_returns = add_constant(factors_returns)

    est = OLS(adjusted_returns, adjusted_market_returns).fit()
    params = est.params.values

    return pd.DataFrame(
        [params, est.tvalues.values, est.pvalues.values],
        index=['value', 't-stat', 'p-value'],
        columns=['alpha'] + [f'beta_{factor}' for factor in factors_returns.iloc[:, 1:]]
    )
