import numpy as np
import pandas as pd

from ..interfaces import IWeighting


class EqualWeights(IWeighting):
    """
    Class for dummy-weighting.
    """

    def weigh(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Weigh values in given data equally.

        Parameters
        ----------
        data : pd.DataFrame
            Data to be weighted. It is implied to get positions (matrix with
            1/0), but it is not obligatory: if data doesn't represent positions
            weights are affected by values of given data.

        Returns
        -------
        pd.DataFrame
            DataFrame with equal weights for given data. It is guaranteed
            that the sum of values in each row is equal to 1.
        """

        weights = np.ones(data.shape, dtype=float)
        weights[np.isnan(data)] = np.nan
        return pd.DataFrame(
            weights / np.nansum(weights, axis=1)[:, np.newaxis],
            index=data.index,
            columns=data.columns
        )
