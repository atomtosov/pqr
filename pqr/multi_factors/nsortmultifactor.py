import numpy as np

from .multifactor import MultiFactor
from pqr.utils import Quantiles


class NSortMultiFactor(MultiFactor):
    """
    Class for multi-factors to pick stocks by intercepting picks of factors.

    Parameters
    ----------
    factors : sequence of Factor
        Sequence of factors.
    name : str, optional
        Name of factor.

    Attributes
    ----------
        dynamic
        bigger_better
        periodicity
        name
        factors
    """

    def pick(self,
             data: np.ndarray,
             interval: Quantiles,
             looking_period: int = 1,
             lag_period: int = 0) -> np.ndarray:
        """
        Pick stocks from data, using some interval.

        Provide the same interface as Factor.pick().

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

        # TODO: check data

        if not isinstance(interval, Quantiles):
            raise ValueError('interval must be Quantiles')

        different_factors = self.bigger_better is None
        for factor in self.factors:
            # update data by picking stocks by interval every time
            data = data * factor.pick(
                data,
                interval if (not different_factors or factor.bigger_better)
                # mirroring quantiles
                else Quantiles(1 - interval.upper, 1 - interval.lower)
            )
            data = data.astype(float)
            data[data == 0] = np.nan
        return ~np.isnan(data)