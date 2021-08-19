"""
This module provides instruments to create benchmarks. Benchmarks treated as theoretical portfolios,
which each investor dreams to beat. Usually there are already good benchmarks - indices
(e.g. S&P 500 or IMOEX). But if for some reason you cannot find suitable benchmark you can build it
from stock universe (with filters and weights, but without selecting stocks). The benchmark will
include all available (filtered) stock universe in each period  without transaction costs, but with
weighting positions (optionally).

In most cases you need a benchmark just to compare its performance with performance of a portfolio,
but do not forget, that portfolios are also can be used as benchmarks for calculating
metrics/plotting the performance. If you want to create benchmark with selecting stocks, just
construct the portfolio.
"""

import pandas as pd

import pqr.portfolios

__all__ = [
    'Benchmark',
]


class Benchmark:
    """
    Class for benchmarks.

    Parameters
    ----------
    name : str, default='benchmark'
        Name of the benchmark.
    """

    name: str
    """Name of the benchmark."""
    returns: pd.Series
    """Periodical returns of the benchmark (non-cumulative)."""

    def __init__(self, name='benchmark'):
        self.name = name

        self.returns = pd.Series()

    def __repr__(self):
        return f'Benchmark({repr(self.name)})'

    def __str__(self):
        return self.name

    def from_index(self, index_values):
        """
        Creates benchmark from existing index (e.g. S&P500 or IMOEX).

        Parameters
        ----------
        index_values : pd.Series
            Series of index values. Percentage changes of these values are used as returns of the
            benchmark.

        Returns
        -------
        Benchmark
            Benchmark with filled returns.
        """

        self.returns = index_values.pct_change()
        self.returns.name = self.name

        return self

    def from_stock_universe(self, stock_prices, mask=None, weighting_factor=None):
        """
        Creates benchmark from stock universe.

        This type of a benchmark should be used, when there is no relevant index to compare with
        portfolios (e.g. if stock universe is too specific or just is severe filtered). In other
        cases it is highly recommended to download data for some existing benchmark, because it is
        really hard to accurately replicate any real stock index.

        To build a benchmark from stock universe you can use filtration and weighting, but not
        picking the stocks. If picking is needed, just construct the portfolio, it also can be used
        as benchmark.

        Parameters
        ----------
        stock_prices : pd.DataFrame
            Prices, representing stock universe. All of available (not np.nan) for selecting stocks
            are picked into the "benchmark-portfolio".
        mask : pd.DataFrame, optional
            Mask to filter stock universe.
        weighting_factor : Factor, optional
            Factor, weighting positions (if you want to replicate some market-cap weighted
            benchmark). If not passed, weighs positions equally.

        Returns
        -------
        Benchmark
            Benchmark with filled returns.
        """

        benchmark_portfolio = pqr.portfolios.Portfolio()
        benchmark_portfolio.pick_all_stocks(stock_prices, mask)
        if weighting_factor is not None:
            benchmark_portfolio.weigh_by_factor(weighting_factor)
        else:
            benchmark_portfolio.weigh_equally()
        benchmark_portfolio.allocate(stock_prices)

        self.returns = benchmark_portfolio.returns
        self.returns.name = self.name

        return self
