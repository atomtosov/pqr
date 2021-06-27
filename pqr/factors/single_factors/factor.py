import numpy as np
import pandas as pd

from .singlefactor import SingleFactor
from ..interfaces import IPicking
from pqr.intervals import Interval, Quantiles, Thresholds
from pqr.utils import epsilon


class Factor(SingleFactor, IPicking):
    """
    Class for factors used to pick stocks.

    Parameters
    ----------
    data : pd.DataFrame
        Matrix with values of factor.
    dynamic : bool, default=False
        Whether factor values should be used to make decisions in absolute form
        or in relative form (percentage changes).
    bigger_better : bool, None, default=True
        Whether more factor value, better company or less factor value better
        company. If it equals None, cannot be defined correctly (e.g. intercept
        multi-factor).
    name : str, optional
        Name of factor.

    Attributes
    ----------
        dynamic
        bigger_better
        name
    """

    def pick(self,
             data: pd.DataFrame,
             interval: Interval,
             looking_period: int = 1,
             lag_period: int = 0) -> pd.DataFrame:
        """
        Pick stocks from data, using some interval.

        Parameters
        ----------
        data : pd.DataFrame
            Data, from which stocks are picked. If some values are missed in
            data but exist in factor values, they are excluded from factor
            values too to prevent situations, when stock cannot be traded, but
            picked.
        interval : Interval
            Interval of factor values to pick. Can be Quantiles, Thresholds or
            Top.
        looking_period : int, default=1
            Looking period to transform factor values
            (see SingleFactor.transform()).
        lag_period : int, default=0
            Lag period to transform factor values
            (see SingleFactor.transform()).

        Returns
        -------
            2-d matrix of bool values. True means that stock is picked, False -
            isn't picked.

        Raises
        ------
        ValueError
            Given data is incorrect or given interval is not supported to pick
            stocks.
        """

        # TODO: check data

        values = self.transform(looking_period, lag_period)
        # exclude values which are not available in data (e.g. after filtering)
        values[data.isna()] = np.nan

        if isinstance(interval, Quantiles):
            lower_threshold = values.quantile(interval.lower, axis=1)
            upper_threshold = values.quantile(interval.upper, axis=1)
            # to include stock with highest factor value
            if interval.upper == 1:
                upper_threshold += epsilon
            choice = (lower_threshold.values[:, np.newaxis] <= values) & \
                     (values < upper_threshold.values[:, np.newaxis])
            data = (data * choice).astype(float)
            data[data == 0] = np.nan
            return ~data.isna()
        elif isinstance(interval, Thresholds):
            choice = (interval.lower <= values) & (values < interval.upper)
            data = (data * choice).astype(float)
            data[data == 0] = np.nan
            return ~data.isna()
        else:
            raise ValueError('interval must be Quantiles or Thresholds')
