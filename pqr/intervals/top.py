from typing import Union

from .interval import Interval


class Top(Interval):
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
    ValueError
        Upper top level more than lower or one of top levels is less than 1.
    TypeError
        A top level isn't int.
    """

    def __init__(self,
                 upper: int = 1,
                 lower: int = 10):

        super().__init__(upper, lower)

        if not (isinstance(self.lower, int) and isinstance(self.upper, int)):
            raise TypeError('top levels must be int')

        if self.lower < 1 or self.upper < 1:
            raise ValueError('top levels must be more than 1')

    @property
    def lower(self) -> Union[int, float]:
        return self._upper

    @property
    def upper(self) -> Union[int, float]:
        return self._lower
