from typing import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm

from pqr.core.benchmark import Benchmark
from pqr.core.portfolio import Portfolio
from pqr.utils import align_many
from .utils import adjust

__all__ = [
    "FamaFrenchRegression",
    "FamaMacbethRegression",
]


class FamaFrenchRegression:
    def __init__(
            self,
            market: Benchmark,
            benchmarks: Sequence[Benchmark],
            rf: float = 0.0
    ):
        self.market = market
        self.benchmarks = benchmarks
        self.rf = rf

    def __call__(self, portfolio: Portfolio) -> pd.DataFrame:
        adjusted_returns = adjust(portfolio.returns, self.rf)
        adjusted_market_returns = adjust(self.market.returns, self.rf)

        y, *x = align_many(
            adjusted_returns,
            adjusted_market_returns,
            *(benchmark.returns for benchmark in self.benchmarks)
        )
        y = y.to_numpy()
        x = sm.add_constant(np.array(x).T)

        ols = sm.OLS(y, x)
        est = ols.fit()

        return pd.DataFrame(
            [
                [
                    est.params[i],
                    est.bse[i],
                    est.tvalues[i],
                    est.pvalues[i],
                ]
                for i in range(len(est.params))
            ],
            index=[
                "alpha",
                f"beta_{self.market.name}",
                *(f"beta_{benchmark.name}" for benchmark in self.benchmarks)
            ],
            columns=[
                "coef",
                "se",
                "t_stat",
                "p_value",
            ]
        )


class FamaMacbethRegression:
    pass
