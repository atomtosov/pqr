from .intervalportfolio import IntervalPortfolio
from pqr.intervals import Quantiles


class QuantilePortfolio(IntervalPortfolio):
    """
    Class for portfolios, using specific quantiles of factor values to pick
    stocks to invest in.

    Parameters
    ----------
    quantiles : Quantiles
        Interval of quantiles, from which stocks are picked.

    Attributes
    ----------
    positions
    returns
    benchmark
    shift
    cumulative_returns
    total_return
    quantiles
    """

    def __init__(self, quantiles: Quantiles):
        """
        Initialize IntervalPortfolio instance.
        """

        if isinstance(quantiles, Quantiles):
            super().__init__(quantiles)
        else:
            raise TypeError('quantiles must be Quantiles')

    @property
    def quantiles(self) -> Quantiles:
        return self._interval
