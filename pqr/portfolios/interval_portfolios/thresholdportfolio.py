from .intervalportfolio import IntervalPortfolio
from pqr.intervals import Thresholds


class ThresholdPortfolio(IntervalPortfolio):
    """
    Class for portfolios, using specific thresholds of factor values to pick
    stocks to invest in.

    Parameters
    ----------
    thresholds : Thresholds
        Interval of thresholds, from which stocks are picked.

    Attributes
    ----------
    positions
    returns
    benchmark
    shift
    cumulative_returns
    total_return
    thresholds
    """

    def __init__(self, thresholds: Thresholds):
        """
        Initialize ThresholdPortfolio instance.
        """

        if isinstance(thresholds, Thresholds):
            super().__init__(thresholds)
        else:
            raise ValueError('thresholds must be Thresholds')

    @property
    def thresholds(self) -> Thresholds:
        return self._interval
