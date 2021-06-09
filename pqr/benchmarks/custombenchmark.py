from typing import Union

import numpy as np
import pandas as pd

from .benchmark import Benchmark
from pqr.factors import WeightingFactor


class CustomBenchmark(Benchmark):
    def __init__(
            self,
            prices: Union[np.ndarray, pd.DataFrame],
            weighting_factor: WeightingFactor = None
    ):
        if weighting_factor is None:
            weighting_factor = WeightingFactor(
                np.ones(prices.shape)
            )
        super().__init__(
            prices * weighting_factor.weigh(
                np.ones(prices.shape)
            )
        )
