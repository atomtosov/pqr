from .intervalportfolio import IntervalPortfolio
from pqr.intervals import Top


class TopNPortfolio(IntervalPortfolio):
    def __init__(self, top: Top):
        if isinstance(top, Top):
            super().__init__(top)
        else:
            raise TypeError('quantiles must be Quantiles')

    @property
    def quantiles(self) -> Top:
        return self._interval
