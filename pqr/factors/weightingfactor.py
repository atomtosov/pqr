import numpy as np

from pqr.base.factor import BaseFactor, WeightingFactorInterface


class WeightingFactor(BaseFactor, WeightingFactorInterface):
    def weigh(self, data: np.ndarray) -> np.ndarray:
        self._check_match_in_shape(data)

        if self.bigger_better:
            weights = self.values * data
            return weights / np.nansum(weights, axis=1)[:, np.newaxis]
        else:
            raise NotImplementedError
