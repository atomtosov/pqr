from abc import ABC
from typing import Sequence, Optional, Union, Tuple

import numpy as np

from ..multifactor import MultiFactor
from ...basefactor import BaseFactor
from ...interfaces import IPicking


class PickingMultiFactor(MultiFactor, ABC, IPicking):
    """
    Abstract base class for multi-factors, which consist of single-factors.

    Parameters
    ----------
    factors : sequence of IPicking
        Sequence of factors, which implement interface of picking factor.
    weights : sequence of int or float, optional
        Sequence of weights. Must have the same length as factors. By default
        equal weights are used.
    name : str, optional
        Name of factor.

    Attributes
    ----------
        dynamic
        bigger_better
        name
        factors
        weights

    Raises
    ------
    ValueError
        If any of given factors doesn't implement interface of picking factor.
    """

    factors: Tuple[IPicking, ...]

    def __init__(self,
                 factors: Sequence[BaseFactor],
                 weights: Optional[Sequence[Union[int, float]]] = None,
                 name: str = ''):
        """
        Initialize PickingMultiFactor instance.
        """

        if np.all([isinstance(factor, IPicking) for factor in factors]):
            super().__init__(factors, name)
        else:
            raise TypeError('all factors must implement IPicking')

        if weights is None:
            self._weights = np.ones(len(factors))
        elif isinstance(weights, Sequence) \
                and np.all([isinstance(w, (int, float)) for w in weights]):
            self._weights = np.array(weights)
        else:
            raise TypeError('weights must be sequence of int or float')

    @property
    def weights(self) -> np.ndarray:
        return self._weights
