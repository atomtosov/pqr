from typing import Union, List

import numpy as np
import pandas as pd

from .factormodel import FactorModel
from pqr.factors import Factor, FilteringFactor, WeightingFactor
from pqr.portfolios import Portfolio, QuantilePortfolio
from pqr.benchmarks import Benchmark
from pqr.utils import make_intervals


class SingleFactorModel(FactorModel):
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
        quantiles = np.take(
            np.linspace(0, 1, n_quantile_portfolios + 1),
            np.arange(n_quantile_portfolios * 2).
            reshape((n_quantile_portfolios, -1)) -
            np.indices((n_quantile_portfolios, 2))[0]
        )
        self._portfolios = [
            QuantilePortfolio(q, budget, fee_rate, fee_fixed).
            construct(
                prices,
                factor,
                holding_period,
                filtering_factor,
                weighting_factor,
                benchmark
            )
            for q in quantiles
        ]

    @property
    def portfolios(self) -> List[Portfolio]:
        return self._portfolios
