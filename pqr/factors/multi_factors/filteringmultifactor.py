from typing import Tuple, Sequence

import numpy as np
import pandas as pd

from .multifactor import MultiFactor
from ..basefactor import BaseFactor
from ..interfaces import IFiltering


class FilteringMultiFactor(MultiFactor, IFiltering):
    """
    Class for filtering some data (e.g. stock universe) by more than 1 factor.
    """

    factors: Tuple[IFiltering, ...]

    def __init__(self,
                 factors: Sequence[BaseFactor],
                 name: str = ''):
        """
        Initialize FilteringMultiFactor instance.

        Parameters
        ----------
        factors : sequence of IFiltering
            Sequence of only filtering factors.
        name : str, optional
            Name of factor.

        Raises
        ------
        TypeError
            Any of factors doesn't implement filtering interface.
        """

        if np.all([isinstance(factor, IFiltering) for factor in factors]):
            super().__init__(factors, name)
        else:
            raise TypeError('all factors must implement IFiltering')

    def filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Filter values in given data by factors, set in the constructor.

        Parameters
        ----------
        data : pd.DataFrame
            Data to be filtered by factors. Expected to get stock prices, but
            is not obligatory.

        Returns
        -------
        pd.DataFrame
            Dataframe with the same data as given, but filled with nans in
            filtered places.
        """

        for factor in self.factors:
            data = factor.filter(data)
        return data
