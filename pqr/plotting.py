"""This module contains all necessary graphics to visualize. Visualization is very
important while assessing performance of a portfolio or a factor model. Some
features or simply errors can be found only by eye.

For now it includes only very simple visualization of results, but it will be
developed in future.
"""

import matplotlib.pyplot as plt

from .metrics import compound_returns, drawdown

__all__ = [
    'plot_compound_returns',
    'plot_underwater',
]


def plot_compound_returns(portfolios, benchmark=None):
    """Plots compound returns of portfolios (optionally with a benchmark).

    Parameters
    ----------
    portfolios : sequence of Portfolio
        Allocated portfolios.
    benchmark : Portfolio or Benchmark
        "Risk-free" alternative for the `portfolios`.
    """

    for portfolio in portfolios:
        compound_returns(portfolio.returns).plot()

    if benchmark is not None:
        start_trading = min(portfolio.returns.index[0] for portfolio in portfolios)
        benchmark_cum_returns = compound_returns(benchmark.returns)
        benchmark_cum_returns = (benchmark_cum_returns[start_trading:] - 
                                 benchmark_cum_returns[start_trading])
        benchmark_cum_returns.plot(color='gray', alpha=0.8)

    plt.title('Portfolio Compound Returns')
    plt.grid()
    plt.legend()


def plot_underwater(portfolios):
    """Plots underwater of portfolios.

    Parameters
    ----------
    portfolios : sequence of Portfolio
        Allocated portfolios.
    """

    for portfolio in portfolios:
        underwater = drawdown(portfolio.returns)
        underwater.plot()

    plt.title('Portfolio Underwater Plot')
    plt.grid()
    plt.legend()
