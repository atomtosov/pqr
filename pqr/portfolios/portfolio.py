from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd

from pqr.factors import Factor, WeightingFactor, FilteringFactor
from pqr.benchmarks import Benchmark
from pqr.utils import lag, pct_change


class Portfolio(ABC):
    _budget: Union[int, float, None]
    _fee_rate: Union[int, float, None]
    _fee_fixed: Union[int, float, None]

    _positions: np.ndarray
    _returns: np.ndarray
    _benchmark: Union[Benchmark, None]

    def __init__(
            self,
            budget: Union[int, float] = None,
            fee_rate: Union[int, float] = None,
            fee_fixed: Union[int, float] = None
    ):
        self.budget = budget
        self.fee_rate = fee_rate
        self.fee_fixed = fee_fixed

        self._positions = np.array([])
        self._returns = np.array([])
        self._benchmark = None

    @abstractmethod
    def _choose_stocks(
            self,
            factor_values: np.ndarray
    ) -> np.ndarray:
        ...

    def construct(
            self,
            stock_prices: Union[np.ndarray, pd.DataFrame],
            factor: Factor,
            holding_period: int = 1,
            filtering_factor: FilteringFactor = None,
            weighting_factor: WeightingFactor = None,
            benchmark: Benchmark = None
    ):
        if isinstance(stock_prices, pd.DataFrame):
            stock_prices = stock_prices.values

        # filter factor values
        filtered_factor = self._filter_stock_universe(
            factor.values,
            filtering_factor
        )
        # construct positions by factor
        self._positions = self._set_holding_period(
            self._choose_stocks(filtered_factor),
            holding_period,
            factor.shift
        )
        # weighting positions
        weighted_positions = self._set_weights(
            self._positions,
            weighting_factor
        )
        # calculate returns
        self._returns = self._calculate_returns(
            stock_prices,
            weighted_positions
        )

        if benchmark is not None:
            self._benchmark = benchmark

        return self

    @staticmethod
    def _filter_stock_universe(
            factor_values: np.ndarray,
            filtering_factor: FilteringFactor = None
    ) -> np.ndarray:
        if filtering_factor is None:
            filtering_factor = FilteringFactor(
                np.ones(factor_values.shape)
            )
        return filtering_factor.filter(factor_values)

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
        # forward filling rows with nans to
        positions = np.take_along_axis(
            positions,
            np.maximum.accumulate(indices_of_rebalancing_periods, axis=0),
            axis=0
        ).astype(bool)
        positions[:shift] = False
        return positions

    @staticmethod
    def _set_weights(
            positions: np.ndarray,
            weighting_factor: WeightingFactor
    ) -> np.ndarray:
        if weighting_factor is None:
            weighting_factor = WeightingFactor(
                np.ones(positions.shape)
            )
        return weighting_factor.weigh(positions)

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
    def returns(self) -> np.ndarray:
        return self._returns
    
    @property
    def total_returns(self) -> np.ndarray:
        return np.nansum(self.returns, axis=1)
    
    @property
    def cumulative_returns(self):
        return np.nancumsum(self.total_returns)

    @property
    def budget(self) -> Union[int, float, None]:
        return self._budget

    @budget.setter
    def budget(self, value: Union[int, float]) -> None:
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
