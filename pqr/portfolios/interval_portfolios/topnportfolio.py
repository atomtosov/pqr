from .intervalportfolio import IntervalPortfolio
from pqr.intervals import Top


class TopNPortfolio(IntervalPortfolio):
    """
    Class for portfolios, using specific top of factor values to pick
    stocks to invest in.

    Parameters
    ----------
    top : Top
        Top, from which stocks are picked.

    Attributes
    ----------
    positions
    returns
    benchmark
    shift
    cumulative_returns
    total_return
    top
    """

    def __init__(self, top: Top):
        if isinstance(top, Top):
            super().__init__(top)
        else:
            raise TypeError('quantiles must be Quantiles')

    @property
    def top(self) -> Top:
        return self._interval
