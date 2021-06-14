from typing import Union

import numpy as np
import pandas as pd

from pqr.base.benchmark import BaseBenchmark
from pqr.base.factor import EqualWeights
from pqr.factors import WeightingFactor
from pqr.utils import pct_change


class CustomBenchmark(BaseBenchmark):
    def __init__(
            self,
            data: Union[np.ndarray, pd.DataFrame],
            weighting_factor: WeightingFactor = None,
            name: str = None
    ):
        if isinstance(data, np.ndarray):
            if data.ndim != 2:
                raise ValueError('data must be 2-dimensional')
            index = np.arange(data.shape[0])
            self._prices = data
        elif isinstance(data, pd.DataFrame):
            index = np.array(data.index)
            self._prices = data.values
        else:
            raise ValueError('data must be np.ndarray or pd.DataFrame')

        if weighting_factor is None:
            weighting_factor = EqualWeights()

        self._weighting_factor = weighting_factor

        super().__init__(index, name)

    def _calc_returns(self) -> np.ndarray:
        return np.nansum(
            pct_change(self._prices) *
            self._weighting_factor.weigh(self._prices),
            axis=1
        )
