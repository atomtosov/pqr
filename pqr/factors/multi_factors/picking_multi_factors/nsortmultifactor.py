import numpy as np
import pandas as pd

from .pickingmultifactor import PickingMultiFactor
from pqr.intervals import Quantiles


class NSortMultiFactor(PickingMultiFactor):
    """
    Class for multi-factors to pick stocks by intercepting picks of factors.

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

        Picking stocks is based on iterative choice of factors. On every
        iteration unpicked choices are deleting from stock universe and this
        data are given to next factor to pick. So, after the last iteration
        choice is ready.

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
