import numpy as np
import pandas as pd

from .singlefactor import SingleFactor
from ..interfaces import IWeighting


class WeightingFactor(SingleFactor, IWeighting):
    """
    Class for factors, weighting some data (e.g. positions).
    """

    def weigh(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Weigh values in given data by factor values.

        Parameters
        ----------
        data : pd.DataFrame
            Data to be weighted. It is implied to get positions (matrix with
            1/0), but it is not obligatory: if data doesn't represent positions
            weights are affected by values of given data.

        Notes
        -----
            Now work only for bigger_better factors with all positive values.

        Returns
        -------
        pd.DataFrame
            DataFrame with weights for given data. It is guaranteed that the
            sum of values in each row is equal to 1.

        Raises
        ------
        NotImplementedError
            If tried to weigh by factor, which is not bigger_better. That
            mechanics is now unsupported, but will be in future versions.
        """

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
