from typing import Union, Any, Iterable

import numpy as np
import pandas as pd

from .singlefactor import SingleFactor
from pqr.utils import Thresholds


class FilteringFactor(SingleFactor):
    """
    Class for filtering factors: filtering factor is used to filter given
    dataset by factor values, if factor values are in Thresholds
    Extends SingleFactor

    Attributes:
        dynamic: bool - is factor dynamic or not, this information is needed
        for future transformation of factor data
        bigger_better: bool | None - is better, when factor value bigger
        (e.g. ROA) or when factor value lower (e.g. P/E); if value is None it
        means that it cannot be said exactly, what is better (used for multi-
        factors)
        periodicity: DataPeriodicity - info about periodicity or discreteness
        of factor data, used for annualization and smth more
        name: str - name of factor
        thresholds: Thresholds - lower and upper threshold to filter (e.g. we
        can have liquidity filter >1 000 000$ daily turnover)

    Methods:
        transform() - returns transformed values of factor data
        with looking_period and lag_period (NOTE: if factor is dynamic,
        real lag = lag_period + 1)

        filter() - filter values in given dataset with respect to factor values
        and thresholds
    """

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
        """
        Initialization of FilteringFactor

        :param data: matrix of factor values
        :param dynamic: is factor dynamic or not
        :param bigger_better: is better, when factor value bigger
        :param data_periodicity: periodicity or discreteness of factor data
        :param replace_with_nan: value, which interpreted as nan in data
        :param name: name of factor
        :param min_threshold: lower threshold of filter
        :param max_threshold: upper threshold of filter

        :raise ValueError if format of data values is incorrect
        """
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
        Filtering values in given dataset by factor values: check if factor
        values falls between thresholds. If not replace it with np.nan

        :param data: given dataset to be filtered
        :return: 2-dimensional matrix with nans in places, where filtering
        condition is not met
        """
        values = self.transform(looking_period=1, lag_period=0)
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
    """
    Simple dummy for FilteringFactor
    Inherits from FilteringFactor to provide factor, which not filter anything

    Attributes:
        dynamic: bool - is factor dynamic or not, this information is needed
        for future transformation of factor data
        bigger_better: bool | None - is better, when factor value bigger
        (e.g. ROA) or when factor value lower (e.g. P/E); if value is None it
        means that it cannot be said exactly, what is better (used for multi-
        factors)
        periodicity: DataPeriodicity - info about periodicity or discreteness
        of factor data, used for annualization and smth more
        name: str - name of factor
        thresholds: Thresholds - lower and upper threshold to filter (e.g. we
        can have liquidity filter >1 000 000$ daily turnover)

    Methods:
        transform() - returns transformed values of factor data
        with looking_period and lag_period (NOTE: if factor is dynamic,
        real lag = lag_period + 1)

        filter() - filter values in given dataset with respect to factor values
        and thresholds
    """
    def __init__(self, shape: Iterable[int]):
        """
        Initialization of NoFilter

        :param shape: shape of dataset to be filtered (actually not)
        """
        super().__init__(np.ones(shape))
