from typing import Union, Any

import numpy as np
import pandas as pd

from pqr.utils import lag, pct_change


class Factor:
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
            bigger_better: bool = True
    ):
        if isinstance(data, np.ndarray):
            assert len(data.shape) == 2, 'data must be 2-dimensional'
            self._data = data.copy()
            self._index = np.arange(data.shape[0])
            self._columns = np.arange(data.shape[1])
        elif isinstance(data, pd.DataFrame):
            self._data = data.values.copy()
            self._index = np.array(data.index)
            self._columns = np.array(data.columns)
        else:
            raise ValueError('data must be numpy.ndarray or pandas.DataFrame')
        self._data[self._data == replace_with_nan] = np.nan

        self.static = static
        self.bigger_better = bigger_better
        self.looking_period = looking_period
        self.lag = lag

    def _transform(self) -> np.ndarray:
        values = self._data.copy()
        if self.static:
            values = lag(
                values,
                self.looking_period + self.lag
            )
        else:
            values = lag(
                pct_change(values, self.looking_period),
                self.lag + 1
            )
        return values

    def _fill_values(self) -> np.ndarray:
        return self._transform()

    def _reset_values(self) -> None:
        self._values = None

    @property
    def values(self) -> np.ndarray:
        if self._values is None:
            self._values = self._fill_values()
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
