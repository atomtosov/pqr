from typing import Union

import numpy as np


__all__ = [
    'Thresholds',
    'Quantiles',
    'Top',
]


class Thresholds:
    """
    Class for thresholds from lower value to upper.

    Parameters
    ----------
    lower : int, float, default=-np.inf
        Lower threshold.
    upper : int, float, default=np.inf
        Upper threshold.

    Raises
    ------
    ValueError
        Lower threshold more than upper.
    TypeError
        A threshold is not int or float.
    """

    lower: Union[int, float]
    upper: Union[int, float]

    def __init__(self, lower: Union[int, float] = -np.inf,
                 upper: Union[int, float] = np.inf):
        if isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
            self.lower = lower
            self.upper = upper
        else:
            raise TypeError('thresholds must be int or float')

        if self.lower > self.upper:
            raise ValueError('lower threshold must be <= upper')

    def __repr__(self) -> str:
        lower_str = f'{self.lower:.2f}' if isinstance(self.lower, float) \
            else str(self.lower)
        upper_str = f'{self.upper:.2f}' if isinstance(self.upper, float) \
            else str(self.upper)

        return f'{type(self).__name__}({lower_str}, {upper_str})'


class Quantiles(Thresholds):
    """
    Class for quantile thresholds.

    Parameters
    ----------
    lower : int, float, default=0
        Lower quantile.
    upper : int, float, default=1
        Upper quantile.

    Raises
    ------
    ValueError
        One of quantiles isn't in range [0,1].
    """

    def __init__(self, lower: Union[int, float] = 0,
                 upper: Union[int, float] = 1):
        super().__init__(lower, upper)
        if not (0 <= self.lower <= 1 and 0 <= self.upper <= 1):
            raise ValueError('quantiles must be in range [0, 1]')


class Top(Thresholds):
    """
    Class for intervals of top levels.

    Parameters
    ----------
    lower : int, default=10
        Lower top level.
    upper : int, float, default=1
        Upper top level.

    Raises
    ------
    TypeError
        Lower or upper boarder is not int.
    ValueError
        One of top levels is less than 1.
    """

    def __init__(self, upper: int = 1, lower: int = 10):
        super().__init__(upper, lower)

        if not (isinstance(self.lower, int) and isinstance(self.upper, int)):
            raise TypeError('top levels must be int')

        if self.lower < 1 or self.upper < 1:
            raise ValueError('top levels must be more than 1')

        self.lower, self.upper = self.upper, self.lower
