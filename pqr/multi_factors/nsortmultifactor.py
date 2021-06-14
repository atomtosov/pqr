import numpy as np

from pqr.base.multi_factor import BaseMultiFactor
from pqr.base.factor import ChoosingFactorInterface
from pqr.base.limits import BaseLimits, Quantiles


class NSortMultiFactor(BaseMultiFactor, ChoosingFactorInterface):
    def choose(self, data: np.ndarray, by: BaseLimits):
        if isinstance(by, Quantiles):
            for factor in self.factors:
                data = data * factor.choose(
                    data,
                    by if factor.bigger_better
                    else Quantiles(1 - by.upper, 1 - by.lower)
                )
                data = data.astype(float)
                data[data == 0] = np.nan
            return ~np.isnan(data)
        elif isinstance(by, BaseLimits):
            raise NotImplementedError
        else:
            raise ValueError('by must be Limits')
