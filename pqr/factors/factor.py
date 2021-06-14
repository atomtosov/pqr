import numpy as np

from pqr.base.factor import BaseFactor, ChoosingFactorInterface
from pqr.base.limits import BaseLimits, Quantiles, Thresholds
from pqr.utils import epsilon


class Factor(BaseFactor, ChoosingFactorInterface):
    def choose(self, data: np.ndarray, by: BaseLimits) -> np.ndarray:
        self._check_match_in_shape(data)

        # exclude values which are not available in data
        values = np.copy(self.values)
        values[np.isnan(data)] = np.nan

        if isinstance(by, Quantiles):
            lower_threshold = np.nanquantile(values, by.lower, axis=1)
            upper_threshold = np.nanquantile(values, by.upper, axis=1)
            # to include stock with highest factor value
            if by.upper == 1:
                upper_threshold += epsilon
            choice = (lower_threshold[:, np.newaxis] <= values) & \
                     (values < upper_threshold[:, np.newaxis])
            data = (data * choice).astype(float)
            data[data == 0] = np.nan
            return ~np.isnan(data)
        elif isinstance(by, Thresholds):
            choice = (by.lower <= values) & (values < by.upper)
            data = (data * choice).astype(float)
            data[data == 0] = np.nan
            return ~np.isnan(data)
        elif isinstance(by, BaseLimits):
            raise NotImplementedError
        else:
            raise ValueError('by must be Quantiles or Thresholds')
