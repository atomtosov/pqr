from typing import Union

import numpy as np
import pandas as pd

from pqr.base.factor_model import BaseFactorModel
from pqr.factors import Factor, FilteringFactor, WeightingFactor
from pqr.portfolios import QuantilePortfolio
from pqr.benchmarks import Benchmark


class SingleFactorModel(BaseFactorModel):
    def fit(
            self,
            prices: Union[np.ndarray, pd.DataFrame],
            factor: Factor,
            holding_period: int = 1,
            filtering_factor: FilteringFactor = None,
            weighting_factor: WeightingFactor = None,
            benchmark: Benchmark = None,
            budget: Union[int, float] = None,
            fee_rate: Union[int, float] = None,
            fee_fixed: Union[int, float] = None,
            n_quantile_portfolios: int = 3
    ) -> None:
        quantiles = self._get_quantiles(n_quantile_portfolios)
        self._portfolios = [
            QuantilePortfolio(q.lower, q.upper, budget, fee_rate, fee_fixed)
            .construct(
                prices,
                factor,
                holding_period,
                filtering_factor,
                weighting_factor,
                benchmark
            )
            for q in quantiles
        ]
