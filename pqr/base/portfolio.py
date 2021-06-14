from abc import ABC
from typing import Union, Dict
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from .factor import (
    ChoosingFactorInterface,
    WeightingFactorInterface, EqualWeights,
    FilteringFactorInterface, NoFilter
)
from .benchmark import BaseBenchmark
from .limits import BaseLimits
from pqr.utils import lag, pct_change


Alpha = namedtuple('Alpha', ['coef', 'p_value'])
Beta = namedtuple('Beta', ['coef', 'p_value'])


class BasePortfolio(ABC):
    _budget: Union[int, float, None]
    _fee_rate: Union[int, float, None]
    _fee_fixed: Union[int, float, None]

    _index: np.ndarray
    _positions: np.ndarray
    _returns: np.ndarray
    _benchmark: Union[BaseBenchmark, None]
    _shift: int
    _annualization_rate: int

    _positions_value: Union[np.ndarray, None]

    def __init__(
            self,
            limits: BaseLimits,
            budget: Union[int, float] = None,
            fee_rate: Union[int, float] = None,
            fee_fixed: Union[int, float] = None
    ):
        if isinstance(limits, BaseLimits):
            self._limits = limits
        else:
            raise ValueError('limits must be Limits')

        self.budget = budget
        self.fee_rate = fee_rate
        self.fee_fixed = fee_fixed

        self._index = np.array([])
        self._positions = np.array([])
        self._returns = np.array([])
        self._benchmark = None
        self._shift = 0
        self._annualization_rate = np.sqrt(12)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self._limits.lower:.2f}, ' \
               f'{self._limits.upper:.2f})'

    def construct(
            self,
            prices: Union[np.ndarray, pd.DataFrame],
            factor: ChoosingFactorInterface,
            holding_period: int = 1,
            filtering_factor: FilteringFactorInterface = None,
            weighting_factor: WeightingFactorInterface = None,
            benchmark: BaseBenchmark = None
    ):
        if isinstance(prices, np.ndarray):
            self._index = np.arange(prices.shape[0])
        elif isinstance(prices, pd.DataFrame):
            self._index = np.array(prices.index)
            prices = prices.values
        else:
            raise ValueError('stock_prices must be numpy.ndarray '
                             'or pandas.DataFrame')
        if filtering_factor is None:
            filtering_factor = NoFilter()
        if weighting_factor is None:
            weighting_factor = EqualWeights()

        self._shift = factor.shift
        self._annualization_rate = np.sqrt(factor.periodicity)

        positions = factor.choose(prices, self._limits)
        filtered_positions = filtering_factor.filter(positions)
        self._positions = self._set_holding_period(
            filtered_positions,
            holding_period,
            factor.shift
        )
        weighted_positions = weighting_factor.weigh(self._positions)
        # calculate returns
        self._returns = self._calculate_returns(
            prices,
            weighted_positions
        )

        if isinstance(benchmark, BaseBenchmark) or benchmark is None:
            self._benchmark = benchmark
        else:
            raise ValueError('benchmark must be Benchmark')

        return self

    @staticmethod
    def _set_holding_period(
            raw_positions: np.ndarray,
            holding_period: int,
            shift: int
    ) -> np.ndarray:
        positions = np.empty(raw_positions.shape)
        positions[::] = np.nan
        # fill positions only for periods of rebalancing
        positions[shift::holding_period] = raw_positions[shift::holding_period]
        # make matrix of indices of rebalancing periods only
        not_rebalancing_periods = np.isnan(positions)
        indices_of_rebalancing_periods = np.where(
            ~not_rebalancing_periods,
            np.arange(positions.shape[0])[:, np.newaxis],
            0
        )
        # forward filling rows with nans by positions of rebalancing periods
        positions = np.take_along_axis(
            positions,
            np.maximum.accumulate(indices_of_rebalancing_periods, axis=0),
            axis=0
        ).astype(bool)
        positions[:shift] = False
        return positions

    @staticmethod
    def _calculate_returns(
            stock_prices: np.ndarray,
            positions: np.ndarray
    ):
        return lag(positions * lag(pct_change(stock_prices), -1))

    @property
    def positions(self) -> np.ndarray:
        return self._positions

    @property
    def budget(self) -> Union[int, float, None]:
        return self._budget

    @budget.setter
    def budget(self, value: Union[int, float, None]) -> None:
        if isinstance(value, (int, float)) and value > 0 \
                or value is None:
            self._budget = value
        else:
            raise ValueError('budget must be int or float and > 0')

    @property
    def fee_rate(self) -> Union[int, float, None]:
        return self._fee_rate

    @fee_rate.setter
    def fee_rate(self, value: Union[int, float, None]) -> None:
        if isinstance(value, (int, float)) or value is None:
            self._fee_rate = value
        else:
            raise ValueError('fee_rate must be int or float')

    @property
    def fee_fixed(self) -> Union[int, float, None]:
        return self._fee_fixed

    @fee_fixed.setter
    def fee_fixed(self, value: Union[int, float, None]) -> None:
        if isinstance(value, (int, float)) or value is None:
            self._fee_fixed = value
        else:
            raise ValueError('fee_fixed must be int or float')

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
            np.nan_to_num(self._benchmark.returns[self._shift+1:])
        )
        est = sm.OLS(self.returns[self._shift+1:], x).fit()
        return Alpha(
            coef=est.params[0],
            p_value=est.pvalues[0]
        )

    @property
    def beta(self) -> Union[Beta, None]:
        if self._benchmark is None:
            return None
        x = sm.add_constant(
            np.nan_to_num(self._benchmark.returns[self._shift+1:])
        )
        est = sm.OLS(self.returns[self._shift+1:], x).fit()
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
            'Alpha, %': self.alpha.coef * 100,
            'Alpha p-value': self.alpha.p_value,
            'Beta': self.beta.coef,
            'Beta p-value': self.beta.p_value,
            'Sharpe Ratio': self.sharpe,
            'Mean Return, %': self.mean_return * 100,
            'Excessive Return, %': self.excessive_return * 100,
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
