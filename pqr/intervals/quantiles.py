from typing import Union

from .interval import Interval


class Quantiles(Interval):
    """
    Class for intervals of quantiles.

    Parameters
    ----------
    lower : int, float, default=0
        Lower quantile.
    upper : int, float, default=1
        Upper quantile.

    Raises
    ------
    ValueError
        Lower quantile more than upper
        or one of quantiles isn't in range [0,1].
    TypeError
        A boarder isn't int or float.
    """

    def __init__(self,
                 lower: Union[int, float] = 0,
                 upper: Union[int, float] = 1):
        """
        Initialize Quantiles instance.
        """

        super().__init__(lower, upper)
        if not (0 <= self.lower <= 1 and 0 <= self.upper <= 1):
            raise ValueError('quantiles must be in range [0, 1]')

    def mirror(self) -> 'Quantiles':
        return Quantiles(1 - self.upper, 1 - self.lower)
