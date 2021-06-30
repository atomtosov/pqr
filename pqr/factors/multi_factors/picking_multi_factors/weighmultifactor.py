import numpy as np
import pandas as pd

from .pickingmultifactor import PickingMultiFactor
from pqr.intervals import Quantiles


class WeighMultiFactor(PickingMultiFactor):
    """
    Class for multi-factors to pick stocks by linearly weighting factors.

    Parameters
    ----------
    factors : sequence of IPicking
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
             data: pd.DataFrame,
             interval: Quantiles,
             looking_period: int = 1,
             lag_period: int = 0) -> pd.DataFrame:
        """
        Pick stocks from data, using some interval.

        Provide the same interface as PickingFactor.pick().

        Picking stocks works like for simple single factor.

        Parameters
        ----------
        data : pd.DataFrame
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

        if not isinstance(interval, Quantiles):
            raise ValueError('interval must be Quantiles')

        factors = np.array(self.transform(looking_period, lag_period))

        factor = np.nansum(
            factors
            * self.weights[:, np.newaxis, np.newaxis],
            axis=0
        ) / np.nansum(self.weights)

        # exclude values which are not available in data (e.g. after filtering)
        factor[np.isnan(data.values) | np.isnan(factors[0])] = np.nan

        lower_threshold = np.nanquantile(factor, interval.lower, axis=1)
        upper_threshold = np.nanquantile(factor, interval.upper, axis=1)
        choice = (lower_threshold[:, np.newaxis] <= factor) & \
                 (factor <= upper_threshold[:, np.newaxis])
        return pd.DataFrame(
            choice,
            index=data.index,
            columns=data.columns
        ).astype(float)
