"""
This module contains all necessary graphics to visualize. Visualization is very
important while assessing performance of a portfolio or a factor model. Some
features or simply errors can be found only by eye.

For now it includes only very simple visualization of results, but it will be
developed in future.
"""

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

__all__ = [
    'plot_cumulative_returns',
]

import pqr.metrics


def plot_cumulative_returns(
        *portfolios_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None
) -> None:
    """
    Plots cumulative returns of `portfolios` and `benchmark`.

    Parameters
    ----------
    portfolios_returns
        Portfolios, which cumulative returns are plotted.
    benchmark_returns
        Benchmark or portfolio to plot cumulative returns "reference point".
    """

    for portfolio_returns in portfolios_returns:
        pqr.metrics.cumulative_returns(portfolio_returns).plot()

    if benchmark_returns is not None:
        start_trading = portfolios_returns[0].index[0]
        benchmark_cum_returns = pqr.metrics.cumulative_returns(
            benchmark_returns)
        benchmark_cum_returns = (benchmark_cum_returns[start_trading:] -
                                 benchmark_cum_returns[start_trading])
        benchmark_cum_returns.plot(color='gray', alpha=0.7)

    plt.title('Portfolio Cumulative Returns', fontsize=25)
    plt.grid()
    plt.legend()
