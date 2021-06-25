import numpy as np

from .pickingmultifactor import PickingMultiFactor
from pqr.utils import Quantiles, epsilon


class WeighMultiFactor(PickingMultiFactor):
    """
    Class for multi-factors to pick stocks by linearly weighting factors.

    Parameters
    ----------
    factors : sequence of IPickingFactor
        Sequence of factors, which implement interface of picking factor.
    weights : sequence of int or float, optional
        Sequence of weights. Must have the same length as factors. By default
        equal weights are used.
    name : str, optional
        Name of factor.

    Attributes
    ----------
        dynamic
        bigger_better
        periodicity
        name
        factors
        weights

    Raises
    ------
    ValueError
        If any of given factors doesn't implement interface of picking factor.
    """

    def pick(self,
             data: np.ndarray,
             interval: Quantiles,
             looking_period: int = 1,
             lag_period: int = 0) -> np.ndarray:
        """
        Pick stocks from data, using some interval.

        Provide the same interface as Factor.pick().

        Picking stocks works like for simple single factor.

        Parameters
        ----------
        data : np.ndarray
            Data, from which stocks are picked. If some values are missed in
            data but exist in factor values, they are excluded from factor
            values too to prevent situations, when stock cannot be traded, but
            picked.
        interval : Interval
            Interval of factor values to pick. Can be only Quantiles.
        looking_period : int, default=1
            Looking period to transform factor values of every factor
            (see SingleFactor.transform()).
        lag_period : int, default=0
            Lag period to transform factor values of every factor
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

        if not isinstance(interval, Quantiles):
            raise ValueError('interval must be Quantiles')

        values = np.nansum(
            self.transform(looking_period, lag_period)
            * self.weights[:, np.newaxis, np.newaxis],
            axis=0
        )
        # exclude values which are not available in data (e.g. after filtering)
        values[np.isnan(data)] = np.nan

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
