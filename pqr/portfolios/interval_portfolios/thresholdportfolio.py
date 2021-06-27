from .intervalportfolio import IntervalPortfolio
from pqr.intervals import Thresholds


class ThresholdPortfolio(IntervalPortfolio):
    def __init__(self, thresholds: Thresholds,):
        if isinstance(thresholds, Thresholds):
            super().__init__(thresholds)
        else:
            raise ValueError('thresholds must be Thresholds')

    @property
    def thresholds(self) -> Thresholds:
        return self._interval
