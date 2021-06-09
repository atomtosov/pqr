from typing import Union, Any

import numpy as np
import pandas as pd

from .factor import Factor


class FilteringFactor(Factor):
    _min_threshold: Union[int, float]
    _max_threshold: Union[int, float]

    def __init__(
            self,
            data: Union[np.ndarray, pd.DataFrame],
            static: bool = True,
            looking_period: int = 1,
            lag: int = 0,
            replace_with_nan: Any = None,
            bigger_better: bool = True,
            min_threshold: Union[int, float] = -np.inf,
            max_threshold: Union[int, float] = np.inf
    ):
        super().__init__(
            data,
            static,
            looking_period,
            lag,
            replace_with_nan,
            bigger_better
        )

        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    def filter(
            self,
            data: np.ndarray
    ) -> np.ndarray:
        if data.shape != self.values.shape:
            raise ValueError('stock_data must match in shape with factor')

        filter_by_factor = (self.values >= self.min_threshold) & \
                           (self.values <= self.max_threshold)
        filtered_stock_data = data * filter_by_factor
        filtered_stock_data[filtered_stock_data == 0] = np.nan
        return filtered_stock_data

    @property
    def min_threshold(self) -> Union[int, float]:
        return self._min_threshold

    @min_threshold.setter
    def min_threshold(self, value: Union[int, float]) -> None:
        if isinstance(value, (int, float)):
            self._min_threshold = value
        else:
            raise ValueError('min_threshold must be int or float')

    @property
    def max_threshold(self) -> Union[int, float]:
        return self._max_threshold

    @max_threshold.setter
    def max_threshold(self, value: Union[int, float]) -> None:
        if isinstance(value, (int, float)) and value >= self.min_threshold:
            self._max_threshold = value
        else:
            raise ValueError('max_threshold must be int or float '
                             'and >= min_threshold')
