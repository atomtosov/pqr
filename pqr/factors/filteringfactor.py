from typing import Union, Any

import numpy as np
import pandas as pd

from pqr.base.factor import BaseFactor, FilteringFactorInterface
from pqr.base.limits import Thresholds


class FilteringFactor(BaseFactor, FilteringFactorInterface):
    _thresholds: Thresholds

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

        self.thresholds = Thresholds(min_threshold, max_threshold)

    def filter(self, data: np.ndarray) -> np.ndarray:
        self._check_match_in_shape(data)

        filter_by_factor = (self.thresholds.lower <= self.values) & \
                           (self.values <= self.thresholds.upper)
        filtered_stock_data = (data * filter_by_factor).astype(float)
        filtered_stock_data[filtered_stock_data == 0] = np.nan
        return ~np.isnan(filtered_stock_data)

    @property
    def thresholds(self) -> Thresholds:
        return self._thresholds

    @thresholds.setter
    def thresholds(self, value: Thresholds):
        if isinstance(value, Thresholds):
            self._thresholds = value
        else:
            raise ValueError('thresholds must be Thresholds')
