from typing import Union

from .interval import Interval


class Quantiles(Interval):
    """
    Class for intervals of quantiles.
    """

    def __init__(self,
                 lower: Union[int, float] = 0,
                 upper: Union[int, float] = 1):
        """
        Initialize Quantiles instance.

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
        """

        super().__init__(lower, upper)
        if not (0 <= self.lower <= 1 and 0 <= self.upper <= 1):
            raise ValueError('quantiles must be in range [0, 1]')

    def mirror(self) -> 'Quantiles':
        """
        Method for creating new "mirrored" quantile.

        The process of mirroring is very simple:
            new_lower = 1 - upper
            new_upper = 1 - lower

        Returns
        -------
        Quantiles
            Mirrored interval of quantiles.
        """

        return Quantiles(1 - self.upper, 1 - self.lower)
