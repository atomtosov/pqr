import numpy as np

from .multifactor import MultiFactor
from pqr.utils import Quantiles


class InterceptMultiFactor(MultiFactor):
    def choose(self,
               data: np.ndarray,
               interval: Quantiles,
               looking_period: int = 1,
               lag_period: int = 0) -> np.ndarray:
        if not isinstance(interval, Quantiles):
            raise ValueError('interval must be Quantiles')
        different_factors = self.bigger_better is None
        factors_choices = np.array(
            [factor.choose(
                data,
                interval if (not different_factors or factor.bigger_better)
                # mirroring quantiles
                else Quantiles(1-interval.upper, 1-interval.lower),
                looking_period,
                lag_period
            )
                for factor in self.factors]
        )
        factors_choices = np.nanprod(factors_choices, axis=0).astype(float)
        data = (data * factors_choices).astype(float)
        data[data == 0] = np.nan
        return ~np.isnan(data)
