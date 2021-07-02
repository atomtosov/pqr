from typing import Optional

import numpy as np
import pandas as pd

from .baseportfolio import BasePortfolio
from .interfaces import IPortfolio
from pqr.factors.interfaces import IPicking, IFiltering, IWeighting
from pqr.benchmarks.interfaces import IBenchmark
from pqr.intervals import Interval
from pqr.factors import NoFilter, EqualWeights
from pqr.benchmarks import CustomBenchmark


class Portfolio(BasePortfolio, IPortfolio):
    """
    Class for portfolios, based on picking stocks, falling within some interval
    (quantiles, thresholds or top).
    """

    def __init__(self, interval: Interval):
        """
        Initialize Portfolio instance.

         Parameters
        ----------
        interval : Interval
            Interval, used to pick stocks by factor values.
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
    def benchmark(self) -> Optional[IBenchmark]:
        return self._benchmark

    @property
    def shift(self) -> int:
        return self._shift

    @property
    def _name(self) -> str:
        return str(self.interval)

    @property
    def interval(self) -> Interval:
        return self._interval

    def invest(self,
               prices: pd.DataFrame,
               picking_factor: IPicking,
               looking_period: int = 1,
               lag_period: int = 0,
               holding_period: int = 1,
               filtering_factor: Optional[IFiltering] = None,
               weighting_factor: Optional[IWeighting] = None,
               benchmark: Optional[IBenchmark] = None) -> None:
        """
        Form positions in each period basing them on factor's picks.

        At first, stock universe is filtered (filtered values are treated as
        missing). Then, picking factor is used to choose stocks to enter
        positions with. In the last step positions are weighted by weighting
        factor (by default equal weights are used) and returns are calculated.

        Parameters
        ----------
        prices : pd.DataFrame
            Dataframe of prices, representing total stock universe in each
            period.
        picking_factor : IPicking
            Factor to pick stocks into portfolio from filtered stock universe.
        looking_period : int, default=1
            Looking period to transform factor values.
        lag_period : int, default=0
            Lag period to transform factor values.
        holding_period : int, default=1
            Holding period of positions. It impacts the frequency of
            rebalancing periods.
        filtering_factor : IFiltering, optional
            Factor to filter stock universe. If not given, stock universe is
            not filtered at all.
        weighting_factor : IWeighting, optional
            Factor to weigh positions in portfolio. If not given, simple equal
            weights are used.
        benchmark : IBenchmark, optional
            Benchmark to calculate some statistical metrics and compare
            portfolio performance with it. If not given, custom benchmark is
            used: in each period all stocks from stock universe (filtered) are
            bought with equal weights.

        Raises
        ------
        TypeError
            One of (factor, filtering_factor, weighting_factor) doesn't
            implement required interface or benchmark is not correct.
        """

        if not isinstance(picking_factor, IPicking):
            raise TypeError('picking_factor must implement IPicking')

        if filtering_factor is None:
            filtering_factor = NoFilter()
        elif not isinstance(filtering_factor, IFiltering):
            raise TypeError('filtering_factor must implement IFiltering')

        if weighting_factor is None:
            weighting_factor = EqualWeights()
        elif not isinstance(weighting_factor, IWeighting):
            raise TypeError('weighting_factor must implement IWeighting')

        self._shift = looking_period + lag_period + picking_factor.dynamic

        filtered_prices = filtering_factor.filter(prices)
        positions = picking_factor.pick(
            filtered_prices,
            self._interval,
            looking_period,
            lag_period
        )
        self._positions = self._set_holding_period(
            positions,
            holding_period
        )
        weighted_positions = weighting_factor.weigh(self._positions)

        self._returns = (
                weighted_positions * prices.pct_change().shift(-1)
        ).shift().sum(axis=1)

        if benchmark is None:
            self._benchmark = CustomBenchmark(filtered_prices)
        elif isinstance(benchmark, IBenchmark):
            self._benchmark = benchmark
        else:
            raise TypeError('benchmark must implement IBenchmark')

    def _set_holding_period(self,
                            raw_positions: pd.DataFrame,
                            holding_period: int = 1) -> pd.DataFrame:
        """
        Method to set holding period of positions in portfolio. Simulates
        rebalancing each "holding_period" periods.

        Parameters
        ----------
        raw_positions : pd.DataFrame
            Picks of factor in each period (in general they are different from
            time to time).
        holding_period : int, default=1
            Number of periods to hold each stock.

        Returns
        -------
        pd.DataFrame
            Dataframe with positions with respect to rebalancing period.

        Raises
        ------
        TypeError
            Given holding_period is not int.
        ValueError
            Holding period less than 1.
        """

        if not isinstance(holding_period, int):
            raise TypeError('holding_period must be int')
        elif holding_period < 1:
            raise ValueError('holding_period must be >= 1')

        positions = np.empty(raw_positions.shape)
        positions[::] = np.nan
        positions[self.shift::holding_period] = \
            raw_positions[self.shift::holding_period]
        return pd.DataFrame(
            positions,
            index=raw_positions.index,
            columns=raw_positions.columns
        ).ffill()
