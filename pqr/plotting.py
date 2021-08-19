"""
This module contains all necessary graphics to visualize. Visualization is very
important while assessing performance of a portfolio or a factor model. Some
features or simply errors can be found only by eye.

For now it includes only very simple visualization of results, but it will be
developed in future.
"""

import matplotlib.pyplot as plt

import pqr.metrics

__all__ = [
    'plot_cumulative_returns',
    'plot_underwater',
]


def plot_cumulative_returns(portfolios, benchmark=None):
    """
    Plots cumulative returns of portfolios (optionally with a benchmark).

    Parameters
    ----------
    portfolios : sequence of Portfolio
        Allocated portfolios.
    benchmark : Portfolio or Benchmark
        "Risk-free" alternative for the `portfolios`.
    """

    for portfolio in portfolios:
        pqr.metrics.cumulative_returns(portfolio.returns).plot()

    if benchmark is not None:
        start_trading = min(portfolio.returns.index[0] for portfolio in portfolios)
        benchmark_cum_returns = pqr.metrics.cumulative_returns(benchmark.returns)
        benchmark_cum_returns = (benchmark_cum_returns[start_trading:] -
                                 benchmark_cum_returns[start_trading])
        benchmark_cum_returns.plot(color='gray', alpha=0.8)

    plt.title('Portfolio Cumulative Returns')
    plt.grid()
    plt.legend()


def plot_underwater(portfolios):
    """
    Plots underwater of portfolios.

    Parameters
    ----------
    portfolios : sequence of Portfolio
        Allocated portfolios.
    """

    for portfolio in portfolios:
        cumsum_returns = portfolio.returns.cumsum()
        underwater = cumsum_returns - cumsum_returns.cummax()
        underwater.plot()

    plt.title('Portfolio Underwater Plot')
    plt.grid()
    plt.legend()
