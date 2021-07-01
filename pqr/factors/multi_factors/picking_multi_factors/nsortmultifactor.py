import numpy as np
import pandas as pd

from .pickingmultifactor import PickingMultiFactor
from pqr.intervals import Quantiles


class NSortMultiFactor(PickingMultiFactor):
    """
    Class for picking multi-factors with the mechanism of n-sorting picks
    of each factor iteratively.
    """

    def pick(self,
             data: pd.DataFrame,
             interval: Quantiles,
             looking_period: int = 1,
             lag_period: int = 0) -> pd.DataFrame:
        """
        Pick stocks from data, representing stock universe in each period of
        time, using one of interval.

        Picking stocks is based on iterative choice of every factor from the
        changing stock universe on every iteration: factor pick some stocks,
        then others (whose wasn't picked) deleted from the stock universe, and
        next factor pick stocks from the new stock universe.

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
        data = data.copy().astype(float)
        choice = np.zeros_like(data)
        for factor in self.factors:
            # update data by picking stocks by interval every time
            choice = factor.pick(
                data,
                interval if (not different_factors or factor.bigger_better)
                else interval.mirror(),
                looking_period,
                lag_period
            )
            data.values[choice.values == 0] = np.nan
        return choice
