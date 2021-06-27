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
    """
    Class for portfolios, based on picking stocks, falling within some interval
    (quantiles, thresholds or top).

    Parameters
    ----------
    interval : Interval
        Interval, used to pick stocks by factor values.

    Attributes
    ----------
    positions
    returns
    benchmark
    shift
    cumulative_returns
    total_return
    """
    def __init__(self, interval: Interval):
        """
        Initialize IntervalPortfolio instance.
        """
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
        return self._returns

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
        """
        Invest relatively in stocks by factor.

        At first, stock universe is filtered, than portfolio is filled from it
        by choices of factor.

        Parameters
        ----------
        prices : pd.DataFrame
            Prices of stock universe, from which stocks are picked into
            portfolio.
        factor : IPicking
            Factor, representing choice of stocks from stock universe. Must
            have data for the same stock universe.
        looking_period : int, default=1
            Looking period to transform factor values.
        lag_period : int, default=0
            Lag period to transform factor values.
        holding_period : int
            Holding period
        filtering_factor : IFiltering, optional
            Factor, filtering stock universe before picking factors. (e.g.
            liquidity). If not given, prices are not filtered at all.
        weighting_factor : IWeighting, optional
            Factor, weighting positions. If not given, simple equal weights are
            used.
        benchmark : BaseBenchmark, optional

        Raises
        ------
        TypeError
            Given benchmark is not instance of BaseBenchmark.
        """

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
        ).shift().sum(axis=1)

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
