from abc import abstractmethod
from collections import namedtuple
from typing import Union, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

from pqr.benchmarks import BaseBenchmark


Alpha = namedtuple('Alpha', ['coef', 'p_value'])
Beta = namedtuple('Beta', ['coef', 'p_value'])


class BasePortfolio:
    """
    Abstract base class for portfolios of stocks.

    Attributes
    ----------
    positions
    returns
    benchmark
    shift
    cumulative_returns
    total_return
    """

    positions: pd.DataFrame
    returns: pd.Series
    benchmark: Optional[BaseBenchmark]
    shift: int
    cumulative_returns: pd.Series
    total_return: Union[int, float]

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self._name})'

    @property
    @abstractmethod
    def positions(self) -> pd.DataFrame:
        ...

    @property
    @abstractmethod
    def returns(self) -> pd.Series:
        ...

    @property
    @abstractmethod
    def benchmark(self) -> Optional[BaseBenchmark]:
        ...

    @property
    @abstractmethod
    def shift(self) -> int:
        ...

    @property
    @abstractmethod
    def _name(self) -> str:
        ...

    @property
    def cumulative_returns(self) -> pd.Series:
        return np.nancumprod(self.returns + 1) - 1

    @property
    def total_return(self) -> Union[int, float]:
        return self.cumulative_returns[-1]

    @property
    def alpha(self) -> Optional[Alpha]:
        if self.benchmark is None:
            return None
        x = sm.add_constant(
            np.nan_to_num(self.benchmark.returns[self.shift+1:])
        )
        est = sm.OLS(self.returns[self.shift+1:], x).fit()
        return Alpha(
            coef=est.params[0],
            p_value=est.pvalues[0]
        )

    @property
    def beta(self) -> Optional[Beta]:
        if self.benchmark is None:
            return None
        x = sm.add_constant(
            np.nan_to_num(self.benchmark.returns[self.shift+1:])
        )
        est = sm.OLS(self.returns[self.shift+1:], x).fit()
        return Beta(
            coef=est.params[1],
            p_value=est.pvalues[1]
        )

    @property
    def sharpe(self) -> Union[int, float]:
        # TODO: add annualization
        return self.returns.mean() / self.returns.std()

    @property
    def mean_return(self) -> Union[int, float]:
        return self.returns.mean()

    @property
    def excessive_return(self) -> Optional[Union[int, float]]:
        if self.benchmark is None:
            return None
        return self.mean_return - np.nanmean(self.benchmark.returns)

    @property
    def mean_volatility(self) -> Union[int, float]:
        return self.returns.std()

    @property
    def benchmark_correlation(self) -> Optional[Union[int, float]]:
        if self.benchmark is None:
            return None
        return np.corrcoef(
            self.returns,
            np.nan_to_num(self.benchmark.returns)
        )[0, 1]

    @property
    def profitable_periods(self) -> Union[int, float]:
        return (self.returns > 0).sum() / np.size(self.returns)

    @property
    def max_drawdown(self) -> Union[int, float]:
        return (
                self.cumulative_returns -
                np.maximum.accumulate(self.cumulative_returns)
        ).min()

    @property
    def stats(self) -> Dict[str, Union[int, float]]:
        return {
            'Alpha, %': self.alpha.coef * 100
            if self.alpha is not None else None,
            'Alpha p-value': self.alpha.p_value
            if self.alpha is not None else None,
            'Beta': self.beta.coef
            if self.beta is not None else None,
            'Beta p-value': self.beta.p_value
            if self.beta is not None else None,
            'Sharpe Ratio': self.sharpe,
            'Mean Return, %': self.mean_return * 100,
            'Excessive Return, %': self.excessive_return * 100
            if self.excessive_return is not None else None,
            'Total Return, %': self.total_return * 100,
            'Volatility, %': self.mean_volatility * 100,
            'Benchmark Correlation': self.benchmark_correlation,
            'Profitable Periods, %': self.profitable_periods * 100,
            'Maximum Drawdown, %': self.max_drawdown * 100
        }

    def plot_cumulative_returns(self, add_benchmark: bool = True):
        plt.plot(self.returns.index,
                 self.cumulative_returns,
                 label=repr(self))
        if add_benchmark and self.benchmark is not None:
            self.benchmark.plot_cumulative_returns()
