from typing import Tuple, Sequence

import numpy as np

from .multifactor import MultiFactor
from ..interfaces import IFilteringFactor


class FilteringMultiFactor(MultiFactor, IFilteringFactor):
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
        periodicity
        name
        factors
    """

    factors: Tuple[IFilteringFactor, ...]

    def __init__(self,
                 factors: Sequence[IFilteringFactor],
                 name: str = None):
        """
        Initialize FilteringMultiFactor instance.
        """

        if not all([isinstance(factor, IFilteringFactor)
                    for factor in factors]):
            raise ValueError('all factors must implement IFilteringFactor')

        # init parent MultiFactor class
        super().__init__(factors, name)

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
