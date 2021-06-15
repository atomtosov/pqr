import numpy as np

from .multifactor import MultiFactor
from pqr.utils import Quantiles


class NSortMultiFactor(MultiFactor):
    def choose(self,
               data: np.ndarray,
               interval: Quantiles,
               looking_period: int = 1,
               lag_period: int = 0) -> np.ndarray:
        if not isinstance(interval, Quantiles):
            raise ValueError('interval must be Quantiles')

        different_factors = self.bigger_better is None
        for factor in self.factors:
            data = data * factor.choose(
                data,
                interval if (not different_factors or factor.bigger_better)
                # mirroring quantiles
                else Quantiles(1-interval.upper, 1-interval.lower)
            )
            data = data.astype(float)
            data[data == 0] = np.nan
        return ~np.isnan(data)
