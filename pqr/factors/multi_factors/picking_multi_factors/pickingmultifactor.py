from abc import ABC
from typing import Sequence, Optional, Union, Tuple

import numpy as np

from ..multifactor import MultiFactor
from ...basefactor import BaseFactor
from ...interfaces import IPicking


class PickingMultiFactor(MultiFactor, IPicking, ABC):
    """
    Abstract base class for picking multi-factors.
    """

    factors: Tuple[IPicking, ...]

    def __init__(self,
                 factors: Sequence[BaseFactor],
                 weights: Optional[Sequence[Union[int, float]]] = None,
                 name: str = ''):
        """
        Initialize PickingMultiFactor instance.

        Parameters
        ----------
        factors : sequence of IPicking
            Sequence of only picking factors.
        weights : sequence of int or float, optional
            Sequence of weights. Must have the same length as factors. By
            default equal weights are used.
        name : str, optional
            Name of factor.

        Raises
        ------
        TypeError
            Any of factors doesn't implement picking interface or given weights
            are not numbers (int or float).
        ValueError
            Given weights are incompatible with given factors.
        """

        if np.all([isinstance(factor, IPicking) for factor in factors]):
            super().__init__(factors, name)
        else:
            raise TypeError('all factors must implement IPicking')

        if weights is None:
            self._weights = np.ones(len(factors))
        elif not np.all([isinstance(w, (int, float)) for w in weights]):
            raise TypeError('weights must be sequence of int or float')
        elif len(weights) != len(factors):
            raise ValueError('weights must have the same length as length '
                             'of factors')
        else:
            self._weights = np.array(weights)

    @property
    def weights(self) -> np.ndarray:
        """
        np.ndarray : array of weights for factors.
        """

        return self._weights
