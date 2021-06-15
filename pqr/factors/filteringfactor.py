from typing import Union, Any, Iterable

import numpy as np
import pandas as pd

from .singlefactor import SingleFactor
from pqr.utils import Thresholds


class FilteringFactor(SingleFactor):
    _thresholds: Thresholds

    def __init__(self,
                 data: Union[np.ndarray, pd.DataFrame],
                 dynamic: bool = False,
                 bigger_better: bool = True,
                 data_periodicity: str = 'monthly',
                 replace_with_nan: Any = None,
                 name: str = None,
                 min_threshold: Union[int, float] = -np.inf,
                 max_threshold: Union[int, float] = np.inf):
        super().__init__(
            data,
            dynamic,
            bigger_better,
            data_periodicity,
            replace_with_nan,
            name
        )

        self.thresholds = Thresholds(min_threshold, max_threshold)

    def filter(self, data: np.ndarray) -> np.ndarray:
        """
        Принимает на вход данные и возвращает те же данные, но в клетках,
        которые не прошли фильтр появляются np.nan

        :param data:
        :return:
        """
        values = self.transform(1, 0)
        filter_by_factor = (self.thresholds.lower <= values) & \
                           (values <= self.thresholds.upper)
        filtered_data = (data * filter_by_factor).astype(float)
        filtered_data[filtered_data == 0] = np.nan
        return filtered_data

    @property
    def thresholds(self) -> Thresholds:
        return self._thresholds

    @thresholds.setter
    def thresholds(self, value: Thresholds):
        if isinstance(value, Thresholds):
            self._thresholds = value
        else:
            raise ValueError('thresholds must be Thresholds')


class NoFilter(FilteringFactor):
    def __init__(self, shape: Iterable[int]):
        super().__init__(np.ones(shape))
