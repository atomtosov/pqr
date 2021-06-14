from typing import Tuple, Iterable

import numpy as np

from pqr.base.multi_factor import BaseMultiFactor
from pqr.base.factor import ChoosingFactorInterface
from pqr.base.limits import BaseLimits, Quantiles


class InterceptMultiFactor(BaseMultiFactor, ChoosingFactorInterface):
    def choose(self, data: np.ndarray, by: BaseLimits):
        # TODO: change bigger_better behavior
        if isinstance(by, Quantiles):
            factors_choices = np.array(
                [factor.choose(data,
                               by if factor.bigger_better
                               else Quantiles(1 - by.upper, 1 - by.lower))
                 for factor in self.factors]
            )
            factors_choices = np.nanprod(factors_choices, axis=0).astype(float)
            data = (data * factors_choices).astype(float)
            data[data == 0] = np.nan
            return ~np.isnan(data)
        elif isinstance(by, BaseLimits):
            raise NotImplementedError
        else:
            raise ValueError('by must be Limits')
