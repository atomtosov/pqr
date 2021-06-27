from typing import Optional

import numpy as np
import pandas as pd

from ..baseportfolio import BasePortfolio
from ..interfaces import IRelativeInvest
from pqr.factors.interfaces import (
    IPicking,
    IFiltering,
    IWeighting
)
from pqr.intervals import Interval
from pqr.factors import NoFilter, EqualWeights
from pqr.benchmarks import BaseBenchmark


class IntervalPortfolio(BasePortfolio, IRelativeInvest):
    def __init__(self, interval: Interval):
        self._positions = pd.DataFrame()
        self._returns = pd.Series()
        self._benchmark = None
        self._shift = 0

        self._interval = interval

    @property
    def positions(self) -> pd.DataFrame:
        return self._positions

    @property
    def returns(self) -> pd.Series:
        return self._returns.sum(axis=1)

    @property
    def benchmark(self) -> Optional['BaseBenchmark']:
        return self._benchmark

    @property
    def shift(self) -> int:
        return self._shift

    @property
    def _name(self) -> str:
        return f'{self._interval.lower:.2f}, {self._interval.upper:.2f}'

    def invest(self,
               prices: pd.DataFrame,
               factor: IPicking,
               looking_period: int = 1,
               lag_period: int = 0,
               holding_period: int = 1,
               filtering_factor: Optional[IFiltering] = None,
               weighting_factor: Optional[IWeighting] = None,
               benchmark: Optional[BaseBenchmark] = None) -> None:
        if filtering_factor is None:
            filtering_factor = NoFilter()
        if weighting_factor is None:
            weighting_factor = EqualWeights()

        self._shift = looking_period + lag_period + factor.dynamic

        filtered_prices = filtering_factor.filter(prices)
        # raw positions
        positions = factor.pick(
            filtered_prices,
            self._interval,
            looking_period,
            lag_period
        )
        # relative positions
        self._positions = self._set_holding_period(
            positions,
            holding_period
        ).astype(float)
        weighted_positions = weighting_factor.weigh(self._positions)

        # calculate returns
        self._returns = (
                weighted_positions * prices.pct_change().shift(-1)
        ).shift()

        if isinstance(benchmark, BaseBenchmark) or benchmark is None:
            self._benchmark = benchmark
        else:
            raise TypeError('benchmark must implement IBenchmark')

    def _set_holding_period(self,
                            raw_positions: pd.DataFrame,
                            holding_period: int = 1) -> pd.DataFrame:
        if not isinstance(holding_period, int) or holding_period < 1:
            raise ValueError('holding_period must be int >= 1')

        positions = np.empty(raw_positions.shape)
        positions[::] = np.nan
        positions[self.shift::holding_period] = \
            raw_positions[self.shift::holding_period]
        return pd.DataFrame(
            positions,
            index=raw_positions.index,
            columns=raw_positions.columns
        ).ffill()
