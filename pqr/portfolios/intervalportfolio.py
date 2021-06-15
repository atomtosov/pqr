from typing import Union, Iterable

import numpy as np
import pandas as pd

from .portfolio import Portfolio
from pqr.utils import lag, pct_change, Interval
from pqr.factors import (
    Factor,
    FilteringFactor, NoFilter,
    WeightingFactor, EqualWeights
)
from pqr.multi_factors import MultiFactor
from pqr.benchmarks import BaseBenchmark


class IntervalPortfolio(Portfolio):
    _interval: Interval

    def __init__(self,
                 interval: Interval,
                 budget: Union[int, float] = None,
                 fee_rate: Union[int, float] = None,
                 fee_fixed: Union[int, float] = None,
                 name: str = None):
        super().__init__(
            budget,
            fee_rate,
            fee_fixed,
            name if name is not None
            else f'{interval.lower:.2f}, {interval.upper:.2f}'
        )

        self.interval = interval

    def construct(self,
                  data: Union[np.ndarray, pd.DataFrame],
                  factor: Union[Factor, MultiFactor],
                  looking_period: int = 1,
                  lag_period: int = 0,
                  holding_period: int = 1,
                  filtering_factor: Union[FilteringFactor,
                                          Iterable[FilteringFactor]] = None,
                  weighting_factor: Union[WeightingFactor,
                                          Iterable[WeightingFactor]] = None,
                  benchmark: BaseBenchmark = None) -> None:
        if isinstance(data, np.ndarray):
            self.index = np.arange(data.shape[0])
            data = np.array(data)
        elif isinstance(data, pd.DataFrame):
            self.index = np.array(data.index)
            data = np.array(data.values)

        if filtering_factor is None:
            filtering_factor = NoFilter(data.shape)
        if weighting_factor is None:
            weighting_factor = EqualWeights(data.shape)

        # save shift for set_holding_period and for metrics calculation
        self._shift = looking_period + lag_period + factor.dynamic
        # save to annualize sharpe ratio
        self._annualization_rate = np.sqrt(factor.periodicity.value)

        data = filtering_factor.filter(data)
        # raw positions
        positions = factor.choose(
            data,
            self.interval,
            looking_period,
            lag_period
        )
        # save relative positions
        self._positions = self._set_holding_period(
            positions,
            holding_period
        ).astype(float)
        weighted_positions = weighting_factor.weigh(self._positions)

        # calculate returns
        if self.budget is None:
            self._returns = lag(weighted_positions * lag(pct_change(data), -1))
        else:
            raise NotImplementedError

        # save benchmark to calculate metrics
        if isinstance(benchmark, BaseBenchmark):
            self._benchmark = benchmark

    def _set_holding_period(self,
                            raw_positions: np.ndarray,
                            holding_period: int = 1) -> np.ndarray:
        if not isinstance(holding_period, int) or holding_period < 1:
            raise ValueError('holding_period must be int >= 1')
        positions = np.empty(raw_positions.shape)
        positions[::] = np.nan
        # fill positions only for periods of rebalancing
        positions[self._shift::holding_period] = \
            raw_positions[self._shift::holding_period]
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
        positions[:self._shift] = False
        return positions

    @property
    def interval(self) -> Interval:
        return self._interval

    @interval.setter
    def interval(self, value: Interval) -> None:
        if isinstance(value, Interval):
            self._interval = value
        else:
            raise ValueError('interval must be Interval')
