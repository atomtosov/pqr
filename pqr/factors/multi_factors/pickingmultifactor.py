from abc import ABC
from typing import Sequence, Optional, Union, Tuple

import numpy as np

from .multifactor import MultiFactor
from ..interfaces import IPickingFactor


class PickingMultiFactor(ABC, MultiFactor, IPickingFactor):
    """
    Abstract base class for multi-factors, which consist of single-factors.

    Parameters
    ----------
    factors : sequence of IPickingFactor
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
        periodicity
        name
        factors
        weights

    Raises
    ------
    ValueError
        If any of given factors doesn't implement interface of picking factor.
    """

    factors: Tuple[IPickingFactor, ...]

    def __init__(self,
                 factors: Sequence[IPickingFactor],
                 weights: Optional[Sequence[Union[int, float]]] = None,
                 name: str = None):
        """
        Initialize WeightingMultiFactor instance.
        """

        if not all([isinstance(factor, IPickingFactor)
                    for factor in factors]):
            raise ValueError('all factors must implement IPickingFactor')

        # init parent ABC
        ABC.__init__(self)
        # init parent MultiFactor
        MultiFactor.__init__(self, factors, name)
        # init parent IPickingFactor
        IPickingFactor.__init__(self)

        self.weights = weights

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    @weights.setter
    def weights(self, value: Optional[Sequence[Union[int, float]]]) -> None:
        self._weights = np.array(value)
