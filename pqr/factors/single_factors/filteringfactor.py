from typing import Union, Any, Iterable

import numpy as np
import pandas as pd

from .singlefactor import SingleFactor
from ..interfaces import IFiltering
from pqr.utils import Thresholds


class FilteringFactor(SingleFactor, IFiltering):
    """
    Class for factors used to filter stock universe.

    Parameters
    ----------
    data : np.ndarray, pd.DataFrame
        Matrix with values of factor.
    dynamic : bool, default=False
        Whether factor values should be used to make decisions in absolute form
        or in relative form (percentage changes).
    bigger_better : bool, None, default=True
        Whether more factor value, better company or less factor value better
        company. If it equals None, cannot be defined correctly (e.g. intercept
        multi-factor).
    periodicity : str, default='monthly'
        Discreteness of factor with respect to one year (e.g. 'monthly' equals
        to 12, because there are 12 trading months in 1 year).
    replace_with_nan: Any, default=None
        Value to be replaced with np.nan in data.
    name : str, optional
        Name of factor.
    min_threshold : int, float, default=-np.inf
        Lower threshold of factor values to filter stock universe.
    max_threshold : int, float, default=np.inf
        Upper threshold of factor values to filter stock universe.

    Attributes
    ----------
        dynamic
        bigger_better
        periodicity
        name
        thresholds
    """

    _thresholds: Thresholds

    def __init__(self,
                 data: Union[np.ndarray, pd.DataFrame],
                 dynamic: bool = False,
                 bigger_better: bool = True,
                 periodicity: str = 'monthly',
                 replace_with_nan: Any = None,
                 name: str = None,
                 min_threshold: Union[int, float] = -np.inf,
                 max_threshold: Union[int, float] = np.inf):
        """
        Initialize FilteringFactor instance.
        """

        # init parent SingleFactor class
        super().__init__(
            data,
            dynamic,
            bigger_better,
            periodicity,
            replace_with_nan,
            name
        )

        self.thresholds = Thresholds(min_threshold, max_threshold)

    def filter(self, data: np.ndarray) -> np.ndarray:
        """
        Filter stock universe by thresholds for factor values.

        Parameters
        ----------
        data : np.ndarray
            Data to be filtered by factor values. Expected to get stock prices,
            but it isn't obligatory.

        Returns
        -------
            2-d matrix with the same data as given, but with replaced with
            np.nan values, which are filtered by factor values and thresholds.

        Raises
        ------
        ValueError
            Given data doesn't match in shape with factor values.
        """

        # TODO: check data

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


class NoFilter(IFiltering):
    """
    Class for dummy-filtering. Used to replace FilteringFactor with factor,
    which doesn't filter anything, but provides the same interface.
    """

    def filter(self, data: np.ndarray) -> np.ndarray:
        return data
