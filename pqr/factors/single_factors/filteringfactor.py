from typing import Union

import numpy as np
import pandas as pd

from .singlefactor import SingleFactor
from ..interfaces import IFiltering
from pqr.intervals import Thresholds


class FilteringFactor(SingleFactor, IFiltering):
    """
    Class for factors, filtering some data (e.g. stock universe).
    """

    thresholds: Thresholds

    def __init__(self,
                 data: pd.DataFrame,
                 dynamic: bool = False,
                 bigger_better: bool = True,
                 min_threshold: Union[int, float] = -np.inf,
                 max_threshold: Union[int, float] = np.inf,
                 name: str = ''):
        """
        Initialize FilteringFactor instance.

        Parameters
        ----------
        data : pd.DataFrame
            Matrix with values of factor. Must be numeric (contains only
            int/float numbers and nans).
        dynamic : bool, default=False
            Is to interpret factor values statically (values itself) or
            dynamically (changes of values).
        bigger_better : bool, default=True
            Is bigger factor values are responsible for more attractive company
            or vice versa.
        min_threshold : int, float, default=-np.inf
            Lower threshold of factor values to filter stock universe.
        max_threshold : int, float, default=np.inf
            Upper threshold of factor values to filter stock universe.
        name : str, optional
            Name of factor.
        """

        super().__init__(
            data,
            dynamic,
            bigger_better,
            name
        )

        self._thresholds = Thresholds(min_threshold, max_threshold)

    def filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Method for filtering stock universe by thresholds for factor values,
        set in the constructor.

        Parameters
        ----------
        data : pd.DataFrame
            Data to be filtered by factor values. Expected to get stock prices,
            but it isn't obligatory.

        Returns
        -------
        pd.DataFrame
            Dataframe with the same data as given, but filled with nans in
            filtered places.
        """

        factor = self.transform(looking_period=1, lag_period=0)
        factor.values[np.isnan(data.values)] = np.nan
        filter_by_factor = (self.thresholds.lower <= factor.values) & \
                           (factor.values <= self.thresholds.upper)
        filtered_values = data.values.copy().astype(float)
        filtered_values[~filter_by_factor] = np.nan
        return pd.DataFrame(
            filtered_values,
            index=data.index,
            columns=data.columns
        )

    @property
    def thresholds(self) -> Thresholds:
        """
        Thresholds : Thresholds for filtering data by factor values (restrict
        only factor values, not data).
        """

        return self._thresholds
