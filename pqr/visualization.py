"""
This module contains all necessary graphics to visualize. Visualization is very
important while assessing performance of a portfolio or a factor model. Some
features or simply errors can be found only by eye.

For now it includes only very simple visualization of results, but it will be
developed in future.
"""

from typing import Optional

import matplotlib.pyplot as plt

__all__ = [
    'plot_cumulative_returns',
]

import pqr.metrics
import pqr.portfolios
import pqr.benchmarks


def plot_cumulative_returns(
        *portfolios: pqr.portfolios.AbstractPortfolio,
        benchmark: Optional[pqr.benchmarks.Benchmark] = None
) -> None:
    """
    Plots cumulative returns of portfolios (optionally with a benchmark).

    Parameters
    ----------
    portfolios
        Portfolios, which cumulative returns are plotted.
    benchmark
        Benchmark or portfolio to plot cumulative returns "reference point".
    """

    for portfolio in portfolios:
        pqr.metrics.cumulative_returns(portfolio).plot()

    if benchmark is not None:
        start_trading = portfolios[0].returns.index[0]
        benchmark_cum_returns = pqr.metrics.cumulative_returns(benchmark)
        benchmark_cum_returns = (benchmark_cum_returns[start_trading:] -
                                 benchmark_cum_returns[start_trading])
        benchmark_cum_returns.plot(color='gray', alpha=0.8)

    plt.title('Portfolio Cumulative Returns', fontsize=25)
    plt.grid()
    plt.legend()
