import numpy as np
import pandas as pd

from .singlefactor import SingleFactor
from ..interfaces import IPicking
from pqr.intervals import Interval, Quantiles, Thresholds, Top


class PickingFactor(SingleFactor, IPicking):
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

        factor = self.transform(looking_period, lag_period)
        # exclude values which are not available in data (e.g. after filtering)
        factor.values[np.isnan(data.values)] = np.nan

        if isinstance(interval, Quantiles):
            lower_threshold = np.nanquantile(
                factor.values,
                interval.lower,
                axis=1
            )[:, np.newaxis]
            upper_threshold = np.nanquantile(
                factor.values,
                interval.upper,
                axis=1
            )[:, np.newaxis]
        elif isinstance(interval, Thresholds):
            lower_threshold = interval.lower
            upper_threshold = interval.upper
        elif isinstance(interval, Top):
            lower_threshold = np.nanmin(
                factor.apply(
                    pd.Series.nlargest, axis=1, n=interval.lower
                ),
                axis=1
            )[:, np.newaxis]
            upper_threshold = np.nanmin(
                factor.apply(
                    pd.Series.nlargest, axis=1, n=interval.upper
                ),
                axis=1
            )[:, np.newaxis]
        else:
            raise ValueError('interval must be some Interval')

        return pd.DataFrame(
            (lower_threshold <= factor.values)
            & (factor.values <= upper_threshold),
            index=data.index,
            columns=data.columns
        )
