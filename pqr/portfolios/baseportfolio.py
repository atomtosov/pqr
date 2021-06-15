from abc import abstractmethod
from collections import namedtuple
from typing import Union, Dict

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from pqr.benchmarks import BaseBenchmark
from pqr.utils import HasNameMixin, HasIndexMixin, DataPeriodicity


Alpha = namedtuple('Alpha', ['coef', 'p_value'])
Beta = namedtuple('Beta', ['coef', 'p_value'])


class BasePortfolio(HasNameMixin, HasIndexMixin):
    _positions: np.ndarray
    _returns: np.ndarray
    _benchmark: Union[None, BaseBenchmark]

    _shift: int
    _annualization_rate: Union[int, float]

    def __init__(self,
                 name: str = None):
        HasNameMixin.__init__(self, name)
        HasIndexMixin.__init__(self)

        self._positions = np.array([])
        self._returns = np.array([])
        self._benchmark = None

        self._shift = 0
        self._annualization_rate = np.sqrt(
            getattr(DataPeriodicity, 'monthly').value
        )

    @abstractmethod
    def construct(self, *args, **kwargs) -> None:
        ...

    @property
    def positions(self) -> np.ndarray:
        return self._positions

    @property
    def returns(self) -> np.ndarray:
        return np.nansum(self._returns, axis=1)

    @property
    def cumulative_returns(self) -> np.ndarray:
        return np.nancumprod(self.returns + 1) - 1

    @property
    def total_return(self) -> Union[int, float]:
        return self.cumulative_returns[-1]

    @property
    def alpha(self) -> Union[Alpha, None]:
        if self._benchmark is None:
            return None
        x = sm.add_constant(
            np.nan_to_num(self._benchmark.returns[self._shift + 1:])
        )
        est = sm.OLS(self.returns[self._shift + 1:], x).fit()
        return Alpha(
            coef=est.params[0],
            p_value=est.pvalues[0]
        )

    @property
    def beta(self) -> Union[Beta, None]:
        if self._benchmark is None:
            return None
        x = sm.add_constant(
            np.nan_to_num(self._benchmark.returns[self._shift + 1:])
        )
        est = sm.OLS(self.returns[self._shift + 1:], x).fit()
        return Beta(
            coef=est.params[1],
            p_value=est.pvalues[1]
        )

    @property
    def sharpe(self) -> Union[int, float]:
        return self.returns.mean() / self.returns.std() * \
               self._annualization_rate

    @property
    def mean_return(self) -> Union[int, float]:
        return self.returns.mean()

    @property
    def excessive_return(self) -> Union[int, float, None]:
        if self._benchmark is None:
            return None
        return self.mean_return - np.nanmean(self._benchmark.returns)

    @property
    def mean_volatility(self) -> Union[int, float]:
        return self.returns.std()

    @property
    def benchmark_correlation(self) -> Union[int, float, None]:
        if self._benchmark is None:
            return None
        return np.corrcoef(
            self.returns,
            np.nan_to_num(self._benchmark.returns)
        )[0, 1]

    @property
    def profitable_periods(self) -> Union[int, float]:
        return (self.returns > 0).sum() / np.size(self.returns)

    @property
    def max_drawdown(self):
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
        plt.plot(self._index, self.cumulative_returns, label=repr(self))
        if add_benchmark and self._benchmark is not None:
            self._benchmark.plot_cumulative_returns(self._shift)

