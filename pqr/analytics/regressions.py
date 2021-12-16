from dataclasses import dataclass
from typing import Sequence, Protocol

import numpy as np
import pandas as pd
import statsmodels.api as sm

from pqr.core.benchmark import Benchmark
from pqr.core.portfolio import Portfolio
from pqr.utils import align_many, adjust

__all__ = [
    "FamaFrenchRegression",
    "FamaMacbethRegression",
]


class SinglePortfolioRegression(Protocol):
    def fit(self, portfolio: Portfolio) -> pd.DataFrame:
        pass


class MultiPortfolioRegression(Protocol):
    def fit(self, portfolios: Sequence[Portfolio]) -> pd.DataFrame:
        pass


@dataclass
class FamaFrenchRegression:
    market: Benchmark
    benchmarks: Sequence[Benchmark]
    rf: float = 0.0

    def fit(self, portfolio: Portfolio) -> pd.DataFrame:
        adjusted_returns = adjust(portfolio.returns, self.rf)
        adjusted_market_returns = adjust(self.market.returns, self.rf)

        y, *x = align_many(
            adjusted_returns,
            adjusted_market_returns,
            *(benchmark.returns for benchmark in self.benchmarks)
        )
        y = y.to_numpy()
        x = sm.add_constant(np.array(x).T)

        est = sm.OLS(y, x).fit()

        idx = ["alpha", f"beta_{self.market.name}",
               *(f"beta_{benchmark.name}" for benchmark in self.benchmarks)]
        cols = ["coef", "se", "t_stat", "p_value"]
        return pd.DataFrame(
            [[est.params[i],
              est.bse[i],
              est.tvalues[i],
              est.pvalues[i]]
             for i in range(len(est.params))],
            index=idx, columns=cols).round(4)


@dataclass
class FamaMacbethRegression:
    market: Benchmark
    benchmarks: Sequence[Benchmark]
    rf: float = 0.0

    def fit(self, portfolios: Sequence[Portfolio]) -> pd.DataFrame:
        betas, portfolios_returns = [], []
        for portfolio in portfolios:
            ff = FamaFrenchRegression(self.market, self.benchmarks, self.rf)
            portfolio_betas = ff.fit(portfolio).iloc[0, 1:].to_numpy()
            portfolios_returns.append(adjust(portfolio.returns, self.rf))
            betas.append(portfolio_betas)
        betas, portfolios_returns = np.array(betas), np.array(portfolios_returns).T

        betas = sm.add_constant(betas)
        lambdas = []
        for y in portfolios_returns:
            est = sm.OLS(y, betas).fit()
            lambdas.append(est.params[1:])
        lambdas = np.array(lambdas)

        idx = [f"lambda_{benchmark.name}" for benchmark in self.benchmarks]
        cols = ["coef", "se"]
        return pd.DataFrame(
            np.array(
                [lambdas.mean(axis=0),
                 lambdas.std(axis=0) / np.sqrt(lambdas.shape[0])]
            ).T,
            index=idx,
            columns=cols).round(4)
