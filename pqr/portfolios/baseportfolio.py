from abc import abstractmethod
from collections import namedtuple
from typing import Union, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

from pqr.benchmarks.interfaces import IBenchmark


Alpha = namedtuple('Alpha', ['coef', 'p_value'])
Beta = namedtuple('Beta', ['coef', 'p_value'])


class BasePortfolio:
    """
    Abstract base class for portfolios of stocks.
    """

    positions: pd.DataFrame
    returns: pd.Series
    benchmark: Optional[IBenchmark]
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
    def benchmark(self) -> Optional[IBenchmark]:
        ...

    @property
    @abstractmethod
    def shift(self) -> int:
        ...

    @property
    @abstractmethod
    def periodicity(self):
        ...

    @property
    @abstractmethod
    def _name(self) -> str:
        ...

    @property
    def cumulative_returns(self) -> pd.Series:
        return (1 + self.returns).cumprod() - 1

    @property
    def total_return(self) -> Union[int, float]:
        return self.cumulative_returns[-1]

    @property
    def max_drawdown(self) -> Union[int, float]:
        cum_returns = self.cumulative_returns
        return (cum_returns - cum_returns.cummax()).min()

    @property
    def alpha(self) -> Union[Alpha, pd.Series]:
        returns = self.returns.values[self.shift+1:]
        benchmark_returns = self.benchmark.returns.values[self.shift+1:]

        x = sm.add_constant(
            np.nan_to_num(benchmark_returns)
        )
        est = sm.OLS(returns, x).fit()
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
        return self.returns.mean() / self.returns.std() * \
               np.sqrt(self.periodicity.value)

    def mean_return(self,
                    moving: bool = False) -> Union[int, float, pd.Series]:
        if moving:
            return self.returns.rolling(self.periodicity.value).mean()

        return self.returns.values.mean()

    def excessive_return(self,
                         moving: bool = False) -> Union[int, float, pd.Series]:
        if moving:
            return self.mean_return(moving=True) \
                   - self.benchmark.returns.rolling(self.periodicity.value)\
                       .mean()

        return self.mean_return() - np.nanmean(self.benchmark.returns)

    def mean_volatility(self,
                        moving: bool = False) -> Union[int, float, pd.Series]:
        if moving:
            return self.returns.rolling(self.periodicity.value).std()

        return self.returns.values.std()

    def benchmark_correlation(self,
                              moving: bool = False) -> Union[int, float,
                                                             pd.Series]:
        if moving:
            return self.returns.rolling(self.periodicity.value)\
                .corr(self.benchmark.returns)

        return self.returns.corr(self.benchmark.returns)

    def profitable_periods(self,
                           moving: bool = False) -> Union[int, float,
                                                          pd.Series]:
        if moving:
            return (self.returns > 0).rolling(self.periodicity.value).sum() / 12

        return (self.returns.values > 0).sum() / np.size(self.returns.values)

    @property
    def stats(self) -> Dict[str, Union[int, float]]:
        return {
            'Alpha, %': self.alpha.coef * 100,
            'Alpha p-value': self.alpha.p_value,
            'Beta': self.beta.coef,
            'Beta p-value': self.beta.p_value,
            'Sharpe Ratio': self.sharpe,
            'Mean Return, %': self.mean_return() * 100,
            'Excessive Return, %': self.excessive_return() * 100
            if self.excessive_return is not None else None,
            'Total Return, %': self.total_return * 100,
            'Volatility, %': self.mean_volatility() * 100,
            'Benchmark Correlation': self.benchmark_correlation(),
            'Profitable Periods, %': self.profitable_periods() * 100,
            'Maximum Drawdown, %': self.max_drawdown * 100
        }

    def plot_cumulative_returns(self, add_benchmark: bool = True):
        plt.plot(self.returns.index,
                 self.cumulative_returns,
                 label=repr(self))
        if add_benchmark and self.benchmark is not None:
            self.benchmark.plot_cumulative_returns(self.shift)
