from typing import Union

import numpy as np


class Interval:
    """
    Class for intervals from lower boarder to upper.

    Parameters
    ----------
    lower : int, float, default=-np.inf
        Lower boarder of interval.
    upper : int, float, default=np.inf
        Upper boarder of interval.

    Raises
    ------
    ValueError
        Lower boarder more than upper.
    TypeError
        A boarder is not int or float.
    """

    lower: Union[int, float]
    upper: Union[int, float]

    def __init__(self,
                 lower: Union[int, float] = -np.inf,
                 upper: Union[int, float] = np.inf):
        """
        Initialize Interval instance.
        """

        if isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
            self._lower = lower
            self._upper = upper
        else:
            raise TypeError('interval boarders must be int or float')

        if self._lower > self._upper:
            raise ValueError('lower interval boarder must be <= upper')

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(lower={self.lower}, ' \
               f'upper={self.upper})'

    @property
    def lower(self) -> Union[int, float]:
        return self._lower

    @property
    def upper(self) -> Union[int, float]:
        return self._upper
