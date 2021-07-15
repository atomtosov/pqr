from typing import Optional

import matplotlib.pyplot as plt

import pqr.benchmarks
import pqr.portfolios


def plot_cumulative_returns(
        *portfolios: pqr.portfolios.AbstractPortfolio,
        benchmark: Optional[pqr.benchmarks.AbstractBenchmark] = None) -> None:
    for portfolio in portfolios:
        plt.plot(portfolio.cumulative_returns, label=repr(portfolio))

    if benchmark is not None:
        shift = min(portfolio.trading_start for portfolio in portfolios)
        cum_returns = (benchmark.cumulative_returns[shift:] -
                       benchmark.cumulative_returns[shift])
        plt.plot(cum_returns, label=repr(benchmark))

    plt.suptitle('Portfolio Cumulative Returns', fontsize=25)
    plt.grid()
    plt.legend()
