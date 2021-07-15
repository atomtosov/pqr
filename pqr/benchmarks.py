import abc
from typing import Optional

import pandas as pd

import pqr.factors

__all__ = [
    'AbstractBenchmark',
    'IndexBenchmark',
    'CustomBenchmark'
]


class AbstractBenchmark(abc.ABC):
    """
    Abstract base class for benchmarks.
    """

    returns: pd.Series

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.returns.name})'

    @property
    def cumulative_returns(self) -> pd.Series:
        return (1 + self.returns).cumprod() - 1


class IndexBenchmark(AbstractBenchmark):
    """
    Class for benchmarks from existing assets indices (e.g. S&P500).

    Parameters
    ----------
    index_values : pd.Series
        Values of some benchmark-index. Percentage changes of that values
        are used as returns of benchmark.
    """

    def __init__(self, index_values: pd.Series):
        self.returns = index_values.pct_change()


class CustomBenchmark(AbstractBenchmark):
    """
    Class for custom benchmarks from stock universe.

    Parameters
    ----------
    prices : pd.DataFrame
        Prices of stocks to be picked. All of available stocks are in the
        portfolio in each period.
    weighting_factor : IWeighting, optional
        Factor to weigh stocks in portfolio (e.g. market capitalization).
        If not passed, equal weights are used.
    """

    def __init__(self,
                 prices: pd.DataFrame,
                 weighting_factor: Optional[pqr.factors.Factor] = None):
        picks = pqr.factors.pick(prices)
        weights = pqr.factors.weigh(picks, weighting_factor)
        universe_returns = prices.pct_change().shift(-1)
        self.returns = (weights * universe_returns).shift().sum(axis=1)
        self.returns.name = ''
