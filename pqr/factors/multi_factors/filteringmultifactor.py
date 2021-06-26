from typing import Tuple, Sequence

import numpy as np

from .multifactor import MultiFactor
from ..basefactor import BaseFactor
from ..interfaces import IFiltering


class FilteringMultiFactor(MultiFactor, IFiltering):
    """
    Class for multi-factors used to filter positions. Consists of factors,
    implementing interface of filtering factors.

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

    factors: Tuple[IFiltering, ...]

    def __init__(self,
                 factors: Sequence[BaseFactor],
                 name: str = ''):
        """
        Initialize FilteringMultiFactor instance.
        """

        if np.all([isinstance(factor, IFiltering) for factor in factors]):
            super().__init__(factors, name)
        else:
            raise ValueError('all factors must implement IFiltering')

    def filter(self, data: np.ndarray) -> np.ndarray:
        """
        Filter values in given data by factors.

        Parameters
        ----------
        data : np.ndarray
            Data to be filtered.

        Returns
        -------
            2-d matrix with filtered data.

        Raises
        ------
        ValueError
            Given data doesn't match in shape with factor values.
        """

        for factor in self.factors:
            data = factor.filter(data)
        return data
