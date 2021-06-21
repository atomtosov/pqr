from typing import Sequence, Union

import numpy as np

from .multifactor import MultiFactor
from pqr.factors import Factor
from pqr.utils import Quantiles, epsilon


class WeightedMultiFactor(MultiFactor):
    """
    Class for multi-factors to pick stocks by weighting factors.

    Parameters
    ----------
    factors : sequence of Factor
        Sequence of factors.
    weights : sequence of int, float
        Sequence of positive weights. If sum of weights more than 1, weights
        are normalizing. If not given, equal weights are used.
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
    """

    _weights: np.ndarray

    def __init__(
            self,
            factors: Sequence[Factor],
            weights: Sequence[Union[int, float]] = None,
            name: str = None
    ):
        """
        Initializing WeightingFactor instance.
        """

        # init parent MultiFactor class
        super().__init__(factors, name)

        self.weights = weights

    def transform(self,
                  looking_period: int = 1,
                  lag_period: int = 1) -> np.ndarray:
        """
        Transform factor values into appropriate for decision-making format.

        At first, all factors are transformed by the same looking and lag
        periods. Then, they are summed up with given weights (by default
        weights are equal).

        Parameters
        ----------
        looking_period : int, default=1
            Period to lookahead in data to transform it.
        lag_period : int, default=0
            Period to lag data to create effect of delayed reaction to factor
            values.

        Returns
        -------
            2-d matrix with shape equal to shape of data with transformed
            factor values. First looking_period+lag_period lines are equal to
            np.nan, because in these moments decision-making is abandoned
            because of lack of data. For dynamic factors one more line is equal
            to np.nan (see above).
        """

        # gather transformed factors
        factors = np.array(
            [factor.transform(looking_period, lag_period)
             for factor in self.factors]
        )
        # weigh factors
        weighted_factors = factors * self.weights[:, np.newaxis, np.newaxis]
        return np.nansum(weighted_factors, axis=0)

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

        values = self.transform(looking_period, lag_period)
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

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value: Union[Sequence[Union[int, float]], None]):
        if value is None:
            self._weights = np.ones(len(self.factors)) / len(self.factors)
        elif np.all([isinstance(w, (int, float)) and w > 0 for w in value]):
            # normalize weights if necessary (sum of weights must be = 1)
            value = np.array(value) / np.sum(value)
            self._weights = value
        else:
            raise ValueError('weights must be Sequence of int or float > 0')
