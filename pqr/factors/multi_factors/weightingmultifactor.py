from typing import Tuple, Sequence

import numpy as np
import pandas as pd

from .multifactor import MultiFactor
from ..basefactor import BaseFactor
from ..interfaces import IWeighting


class WeightingMultiFactor(MultiFactor, IWeighting):
    """
    Class for weighting some data (e.g. positions) by more than 1 factor.
    """

    factors: Tuple[IWeighting, ...]

    def __init__(self,
                 factors: Sequence[BaseFactor],
                 name: str = ''):
        """
        Initialize WeightingMultiFactor instance.

        Parameters
        ----------
        factors : sequence of IWeighting
            Sequence of only weighting factors.
        name : str, optional
            Name of factor.

        Raises
        ------
        TypeError
            Any of factors doesn't implement weighting interface.
        """

        if np.all([isinstance(factor, IWeighting) for factor in factors]):
            super().__init__(factors, name)
        else:
            raise TypeError('all factors must implement IWeighting')

    def weigh(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Weigh values in given data by factors, set in the constructor.

        Parameters
        ----------
        data : pd.DataFrame
            Data to be weighted. Expected positions (matrix with 1/0), but not
            obligatory: if data doesn't represent positions weights are
            affected by values of given data.

        Notes
        -----
            Now work only for bigger_better factors with all positive values.

        Returns
        -------
        pd.DataFrame
            Dataframe with weights for given data. It is guaranteed that the
            sum of values in each row is equal to 1.
        """

        for factor in self.factors:
            data = factor.weigh(data)
        return data
