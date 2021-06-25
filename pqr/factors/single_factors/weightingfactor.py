from typing import Iterable

import numpy as np

from .singlefactor import SingleFactor


class WeightingFactor(SingleFactor):
    """
    Class for factors used to weigh positions.

    Parameters
    ----------
    data : np.ndarray, pd.DataFrame
        Matrix with values of factor.
    dynamic : bool, default=False
        Whether factor values should be used to make decisions in absolute form
        or in relative form (percentage changes).
    bigger_better : bool, None, default=True
        Whether more factor value, better company or less factor value better
        company. If it equals None, cannot be defined correctly (e.g. intercept
        multi-factor).
    periodicity : str, default='monthly'
        Discreteness of factor with respect to one year (e.g. 'monthly' equals
        to 12, because there are 12 trading months in 1 year).
    replace_with_nan: Any, default=None
        Value to be replaced with np.nan in data.
    name : str, optional
        Name of factor.

    Attributes
    ----------
        dynamic
        bigger_better
        periodicity
        name
    """

    def weigh(self, data: np.ndarray) -> np.ndarray:
        """
        Weigh values in given data by factor values.

        Parameters
        ----------
        data : np.ndarray
            Data to be weighted. Expected positions (matrix with True/False or
            1/0), but not obligatory.

        Notes
        -----
            Now work only for bigger_better factors with all positive values.

        Returns
        -------
            2-d matrix with weighted factor values.

        Raises
        ------
        ValueError
            Given data is incorrect or given interval is not supported to pick
            stocks.
        """

        # TODO: check data

        values = self.transform(looking_period=1, lag_period=0)
        if self.bigger_better:
            weights = values * data
            return weights / np.nansum(weights, axis=1)[:, np.newaxis]
        else:
            raise NotImplementedError('lower_better factors '
                                      'are not supported yet')


class EqualWeights(WeightingFactor):
    """
    Class for dummy-weighting. Used to replace WeightingFactor with factor,
    which weigh equally.

    Parameters
    ----------
    shape: iterable of int
        Shape of data to be weighted.

    Attributes
    ----------
        dynamic
        bigger_better
        periodicity
        name
        thresholds
    """

    def __init__(self, shape: Iterable[int]):
        """
        Initialization EqualWeights instance.

        Creates WeightingFactor with matrix, filled with ones to weigh equally.
        """

        super().__init__(np.ones(shape))
