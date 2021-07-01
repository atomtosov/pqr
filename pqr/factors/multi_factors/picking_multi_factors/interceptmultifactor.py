import pandas as pd

from .pickingmultifactor import PickingMultiFactor
from pqr.intervals import Quantiles


class InterceptMultiFactor(PickingMultiFactor):
    """
    Class for picking multi-factors with the mechanism of intercepting picks
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

        Picking stocks is based on choices of every factor by the same data and
        the same interval (if factors are different (with respect to
        bigger_better value), interval is mirrored for lower_better factors).
        Then, matrices with choices are multiplied element-wise (to intercept
        picks of each factor).

        Notes
        -----
            Now supported only picks by quantiles with no use of weights.
            Other intervals and weights will be added in future versions.

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

        different_factors = self.bigger_better is None
        # every factor pick stocks by interval from the same data
        choice = None
        for factor in self.factors:
            pick = factor.pick(
                data,
                # mirroring quantiles
                interval if (not different_factors or factor.bigger_better)
                else interval.mirror(),
                looking_period,
                lag_period
            )
            if choice is None:
                choice = pick.values
            else:
                choice *= pick.values
        return pd.DataFrame(
            choice,
            index=data.index,
            columns=data.columns
        ).astype(float)
