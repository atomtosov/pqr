"""
This module contains all necessary graphics to visualize. Visualization is very
important while assessing performance of a portfolio or a factor model. Some
features or simply errors can be found only by eye.

For now it includes only very simple visualization of results, but it will be
developed in future.
"""


from typing import Optional, Union

import matplotlib.pyplot as plt

import pqr.benchmarks
import pqr.portfolios


__all__ = [
    'plot_cumulative_returns',
]


def plot_cumulative_returns(
        *portfolios: pqr.portfolios.Portfolio,
        benchmark: Optional[Union[pqr.benchmarks.Benchmark,
                                  pqr.portfolios.Portfolio]] = None) -> None:
    """
    Plots cumulative returns of `portfolios` and `benchmark`.

    Parameters
    ----------
    portfolios
        Portfolios, which cumulative returns are plotted.
    benchmark
        Benchmark or portfolio to plot cumulative returns "reference point".
    """

    for portfolio in portfolios:
        plt.plot(portfolio.cumulative_returns, label=str(portfolio))

    if benchmark is not None:
        start_trading = portfolios[0].returns.index[0]
        cum_returns = (benchmark.cumulative_returns[start_trading:] -
                       benchmark.cumulative_returns[start_trading])
        plt.plot(cum_returns, label=str(benchmark), color='gray')

    plt.suptitle('Portfolio Cumulative Returns', fontsize=25)
    plt.grid()
    plt.legend()
