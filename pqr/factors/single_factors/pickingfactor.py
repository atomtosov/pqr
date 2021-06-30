import numpy as np
import pandas as pd

from .singlefactor import SingleFactor
from ..interfaces import IPicking
from pqr.intervals import Interval, Quantiles, Thresholds, Top


class PickingFactor(SingleFactor, IPicking):
    """
    Class for factors, picking stocks from stock universe.
    """

    def pick(self,
             data: pd.DataFrame,
             interval: Interval,
             looking_period: int = 1,
             lag_period: int = 0) -> pd.DataFrame:
        """
        Pick stocks from data, representing stock universe in each period of
        time, using one of interval.

        If the Quantiles are passed as an interval, it simply selects the
        companies whose values fall into the transmitted Quantiles. If given
        interval is Thresholds just pick companies whose factor values fall
        into these Thresholds. If Top is used as interval, simply pick top
        stocks from lower to upper.

        Parameters
        ----------
        data : pd.DataFrame
            Data, representing stock universe (implied that this data is
            prices). If some values are missed in data but exist in factor
            values, they are excluded from factor values too to prevent
            situations, when stock cannot be actually traded, but picked.
        interval : Interval
            Interval of factor values to pick. Can be Quantiles, Thresholds or
            Top.
        looking_period : int, default=1
            Looking back period.
        lag_period : int, default=0
            Delaying period to entry into positions.

        Returns
        -------
        pd.DataFrame
            DataFrame of 1/0, where 1 shows that a stock is fall into given
            interval and should be picked into portfolio, and 0 shows that
            a stock shouldn't.

        Raises
        ------
        ValueError
            Given interval is not supported to pick stocks.
        """

        factor = self.transform(looking_period, lag_period)
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
            raise ValueError('interval must be one of '
                             'Quantiles, Thresholds, Top')

        return pd.DataFrame(
            (lower_threshold <= factor.values)
            & (factor.values <= upper_threshold),
            index=data.index,
            columns=data.columns
        ).astype(float)
