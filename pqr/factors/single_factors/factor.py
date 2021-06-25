import numpy as np

from .singlefactor import SingleFactor
from ..interfaces import IPickingFactor
from pqr.utils import epsilon, Interval, Quantiles, Thresholds


class Factor(SingleFactor, IPickingFactor):
    """
    Class for factors used to pick stocks.

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

    Attributes
    ----------
        dynamic
        bigger_better
        periodicity
        name
    """

    def pick(self,
             data: np.ndarray,
             interval: Interval,
             looking_period: int = 1,
             lag_period: int = 0) -> np.ndarray:
        """
        Pick stocks from data, using some interval.

        Parameters
        ----------
        data : np.ndarray
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
        values[np.isnan(data)] = np.nan

        if isinstance(interval, Quantiles):
            lower_threshold = np.nanquantile(values, interval.lower, axis=1)
            upper_threshold = np.nanquantile(values, interval.upper, axis=1)
            # to include stock with highest factor value
            if interval.upper == 1:
                upper_threshold += epsilon
            choice = (lower_threshold[:, np.newaxis] <= values) & \
                     (values < upper_threshold[:, np.newaxis])
            data = (data * choice).astype(float)
            data[data == 0] = np.nan
            return ~np.isnan(data)
        elif isinstance(interval, Thresholds):
            choice = (interval.lower <= values) & (values < interval.upper)
            data = (data * choice).astype(float)
            data[data == 0] = np.nan
            return ~np.isnan(data)
        else:
            raise ValueError('interval must be Quantiles or Thresholds')
