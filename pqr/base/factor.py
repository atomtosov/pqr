from abc import ABC, abstractmethod
from typing import Union, Any
from enum import Enum

import numpy as np
import pandas as pd

from .limits import BaseLimits
from pqr.utils import lag, pct_change


class DataPeriodicity(Enum):
    monthly = 12
    daily = 252


class BaseFactor(ABC):
    _values: Union[np.ndarray, None]
    _index: np.ndarray
    _columns: np.ndarray

    _static: bool
    _looking_period: int
    _lag: int
    _bigger_better: bool

    def __init__(
            self,
            data: Union[np.ndarray, pd.DataFrame],
            static: bool = True,
            looking_period: int = 1,
            lag: int = 0,
            replace_with_nan: Any = None,
            bigger_better: bool = True,
            data_periodicity: str = 'monthly',
    ):
        if isinstance(data, np.ndarray):
            assert len(data.shape) == 2, 'data must be 2-dimensional'
            self._data = data
            self._index = np.arange(data.shape[0])
            self._columns = np.arange(data.shape[1])
        elif isinstance(data, pd.DataFrame):
            self._data = data.values
            self._index = np.array(data.index)
            self._columns = np.array(data.columns)
        else:
            raise ValueError('data must be numpy.ndarray or pandas.DataFrame')
        self._data[self._data == replace_with_nan] = np.nan
        self._periodicity = getattr(DataPeriodicity, data_periodicity)

        self.static = static
        self.bigger_better = bigger_better
        self.looking_period = looking_period
        self.lag = lag

    def _fill_values(self) -> None:
        self._values = self._data
        if self.static:
            self._values = lag(
                self.values,
                self.looking_period + self.lag
            )
        else:
            self._values = lag(
                pct_change(self.values, self.looking_period),
                self.lag + 1
            )

    def _reset_values(self) -> None:
        self._values = None

    def _check_match_in_shape(self, data: np.ndarray) -> bool:
        if data.shape != self.values.shape:
            raise ValueError('data and factor must match in shape')
        return True

    @property
    def values(self) -> np.ndarray:
        if self._values is None:
            self._fill_values()
        return self._values

    @property
    def index(self) -> np.ndarray:
        return self._index

    @property
    def columns(self) -> np.ndarray:
        return self._columns

    @property
    def static(self) -> bool:
        return self._static

    @static.setter
    def static(self, value: bool) -> None:
        if isinstance(value, bool):
            self._static = value
            self._reset_values()
        else:
            raise ValueError('static must be bool (True or False)')

    @property
    def looking_period(self) -> int:
        return self._looking_period

    @looking_period.setter
    def looking_period(self, value: int) -> None:
        if isinstance(value, int) and value > 0:
            self._looking_period = value
            self._reset_values()
        else:
            raise ValueError('looking_period must be int and > 0')

    @property
    def lag(self) -> int:
        return self._lag

    @lag.setter
    def lag(self, value: int) -> None:
        if isinstance(value, int) and value >= 0:
            self._lag = value
            self._reset_values()
        else:
            raise ValueError('lag must be int and >= 0')

    @property
    def shift(self) -> int:
        return self.looking_period + self.lag + (not self.static)

    @property
    def bigger_better(self) -> bool:
        return self._bigger_better

    @bigger_better.setter
    def bigger_better(self, value: bool) -> None:
        if isinstance(value, bool):
            self._bigger_better = value
        else:
            raise ValueError('bigger_better must be bool (True or False)')
        
    @property
    def periodicity(self) -> int:
        return self._periodicity.value


class ChoosingFactorInterface(ABC):
    @abstractmethod
    def choose(self, data: np.ndarray, by: BaseLimits):
        ...

    @property
    @abstractmethod
    def shift(self) -> int:
        ...

    @property
    @abstractmethod
    def periodicity(self) -> int:
        ...


class FilteringFactorInterface(ABC):
    @abstractmethod
    def filter(self, data: np.ndarray) -> np.ndarray:
        ...


class NoFilter(FilteringFactorInterface):
    def filter(self, data: np.ndarray) -> np.ndarray:
        return data


class WeightingFactorInterface(ABC):
    @abstractmethod
    def weigh(self, data: np.ndarray) -> np.ndarray:
        ...


class EqualWeights(WeightingFactorInterface):
    def weigh(self, data: np.ndarray) -> np.ndarray:
        weights = np.ones(data.shape)
        weights[np.isnan(data) | data == 0] = 0
        return weights / np.nansum(weights, axis=1)[:, np.newaxis]
