"""
This module provides instruments to create benchmarks. Benchmarks treated as
theoretical portfolios, which each investor dreams to beat. Usually there are
already good benchmarks - indices (e.g. S&P 500 or IMOEX). But if for some
reason you cannot find suitable benchmark you can build it from stock universe
(with filters and weights, but without selecting stocks). The benchmark will
include all available (filtered) stock universe in each period  without
transaction costs, but with weighting positions (optionally).

In most cases you need a benchmark just to compare its performance with
performance of a portfolio, but do not forget, that portfolios are also can be
used as benchmarks for calculating metrics/plotting the performance. If you
want to create benchmark with selecting stocks, just construct the portfolio.

"""


import dataclasses
from typing import Optional

import pandas as pd

import pqr.factors
import pqr.thresholds


__all__ = [
    'Benchmark',
    'benchmark_from_index',
    'benchmark_from_stock_universe',
]


@dataclasses.dataclass(frozen=True, repr=False)
class Benchmark:
    """
    Class for theoretical benchmarks.

    Parameters
    ----------
    returns
        Period-to-period returns of a theoretical benchmark.
    """

    returns: pd.Series
    """Period-to-period returns of the benchmark."""

    def __str__(self) -> str:
        return str(self.returns.name)

    @property
    def cumulative_returns(self) -> pd.Series:
        """Cumulative returns of the benchmark."""

        return (1 + self.returns).cumprod() - 1


def benchmark_from_index(index_values: pd.Series) -> Benchmark:
    """
    Creates benchmark from existing index (e.g. S&P500 or IMOEX).

    Parameters
    ----------
    index_values
        Series of index values. Percentage changes of these values are used as
        returns of the benchmark. Name of the series is used as name for the
        benchmark.
    """

    return Benchmark(index_values.pct_change())


def benchmark_from_stock_universe(
        stock_prices: pd.DataFrame,
        filtering_factor: Optional[pd.DataFrame] = None,
        filtering_thresholds:
        pqr.thresholds.Thresholds = pqr.thresholds.Thresholds(),
        weighting_factor: Optional[pd.DataFrame] = None,
        weighting_factor_is_bigger_better: bool = True,
) -> Benchmark:
    """
    Creates custom benchmark from stock universe.

    This type of benchmark should be used, when there is no relevant index to
    compare with portfolios (e.g. if stock universe is too specific or just
    is severe filtrated). In other cases it is highly recommended to download
    data for existing benchmark, because it is really hard to accurately 
    replicate any real stock index with factors.
    
    To build benchmark from stock universe you can use filtration and 
    weighting, but not selecting the stocks. If selecting is needed, just 
    construct the portfolio, it also can be used as benchmark.

    Parameters
    ----------
    stock_prices
        Prices, representing stock universe. All of available (not np.nan) for 
        selecting stocks are picked into the "benchmark-portfolio".
    filtering_factor
        Factor, filtering stock universe. If not given, just do not filter at
        all.
    filtering_thresholds
        Thresholds, limiting `filtering_factor`.
    weighting_factor
        Factor, weighting positions (if you want to replicate some market-cap
        weighted benchmark). If not passed, weighs positions equally.
    weighting_factor_is_bigger_better
        Whether bigger values of `weighting_factor` will lead to bigger weights 
        for a position or on the contrary to lower.
    """

    stock_universe = pqr.factors.filtrate(stock_prices, filtering_factor,
                                          filtering_thresholds)
    picks = pqr.factors.select(stock_universe)
    weights = pqr.factors.weigh(picks, weighting_factor,
                                weighting_factor_is_bigger_better)
    universe_returns = stock_prices.pct_change().shift(-1)
    benchmark_returns = (weights * universe_returns).shift().sum(axis=1)
    benchmark_returns.name = 'benchmark'
    return Benchmark(benchmark_returns)
