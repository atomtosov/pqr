import numpy as np

from pqr.base.multi_factor import BaseMultiFactor
from pqr.base.factor import FilteringFactorInterface


class FilteringMultiFactor(BaseMultiFactor, FilteringFactorInterface):
    def filter(self, data: np.ndarray) -> np.ndarray:
        filter_by_factors = np.array(
            [factor.filter(data) for factor in self.factors]
        )
        filter_by_factors = np.nanprod(filter_by_factors, axis=0)
        filter_by_factors = (data * filter_by_factors).astype(float)
        filter_by_factors[filter_by_factors == 0] = np.nan
        return ~np.isnan(filter_by_factors)
