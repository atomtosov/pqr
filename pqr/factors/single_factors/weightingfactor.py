import numpy as np
import pandas as pd

from .singlefactor import SingleFactor
from ..interfaces import IWeighting


class WeightingFactor(SingleFactor, IWeighting):
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
    name : str, optional
        Name of factor.

    Attributes
    ----------
        dynamic
        bigger_better
        name
    """

    def weigh(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Weigh values in given data by factor values.

        Parameters
        ----------
        data : pd.DataFrame
            Data to be weighted. Expected positions (matrix with True/False or
            1/0), but not obligatory.

        Notes
        -----
            Now work only for bigger_better factors with all positive values.

        Returns
        -------
            2-d matrix with weighted data.

        Raises
        ------
        ValueError
            Given data doesn't match in shape with factor values.
        """

        # TODO: check data

        factor = self.transform(looking_period=1, lag_period=0)
        if self.bigger_better:
            weights = factor.values * data.values
            return pd.DataFrame(
                weights / np.nansum(weights, axis=1)[:, np.newaxis],
                index=data.index,
                columns=data.columns
            )
        else:
            raise NotImplementedError('lower_better factors '
                                      'are not supported yet')


class EqualWeights(IWeighting):
    """
    Class for dummy-weighting. Used to replace WeightingFactor with factor,
    which weigh equally, but provides the same interface.
    """

    def weigh(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            data.values / np.nansum(data.values, axis=1)[:, np.newaxis],
            index=data.index,
            columns=data.columns
        )
