import numpy as np
import pandas as pd

from .pickingmultifactor import PickingMultiFactor
from pqr.intervals import Quantiles


class WeighMultiFactor(PickingMultiFactor):
    """
    Class for picking multi-factors with the mechanism of weighting values
    of each factor.
    """

    def pick(self,
             data: pd.DataFrame,
             interval: Quantiles,
             looking_period: int = 1,
             lag_period: int = 0) -> pd.DataFrame:
        """
        Pick stocks from data, representing stock universe in each period of
        time, using one of interval.

        Picking stocks is based on weighting factor values. Then the matrix of
        weighted factor values is used like a simple single-factor.

        Notes
        -----
            Now supported only picks by quantiles.
            Other intervals will be added in future versions.

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
