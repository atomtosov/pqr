from .intervalportfolio import IntervalPortfolio
from pqr.intervals import Quantiles


class QuantilePortfolio(IntervalPortfolio):
    def __init__(self, quantiles: Quantiles):
        if isinstance(quantiles, Quantiles):
            super().__init__(quantiles)
        else:
            raise TypeError('quantiles must be Quantiles')

    @property
    def quantiles(self) -> Quantiles:
        return self._interval
