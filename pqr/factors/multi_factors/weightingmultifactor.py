from typing import Tuple, Sequence

import numpy as np
import pandas as pd

from .multifactor import MultiFactor
from ..basefactor import BaseFactor
from ..interfaces import IWeighting


class WeightingMultiFactor(MultiFactor, IWeighting):
    """
    Class for multi-factors used to weigh positions. Consists of factors,
    implementing interface of weighting factors.

    Parameters
    ----------
    factors : sequence of IFactor
        Sequence of factors.
    name : str, optional
        Name of factor.

    Attributes
    ----------
        dynamic
        bigger_better
        name
        factors
    """

    factors: Tuple[IWeighting, ...]

    def __init__(self,
                 factors: Sequence[BaseFactor],
                 name: str = ''):
        """
        Initialize WeightingMultiFactor instance.
        """

        if np.all([isinstance(factor, IWeighting) for factor in factors]):
            super().__init__(factors, name)
        else:
            raise TypeError('all factors must implement IWeighting')

    def weigh(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Weigh values in given data by factors.

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

        for factor in self.factors:
            data = factor.weigh(data)
        return data
