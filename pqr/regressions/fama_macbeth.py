__all__ = [
    "estimate_fama_macbeth",
]

from typing import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm

from pqr.regressions.fama_french import estimate_fama_french
from pqr.utils import (
    adjust,
    align,
)


def estimate_fama_macbeth(
        portfolios: Sequence[pd.DataFrame],
        market: pd.Series,
        benchmarks: Sequence[pd.Series],
        rf: float = 0.0,
) -> pd.DataFrame:
    # 1st step: estimate betas
    betas = []
    for portfolio in portfolios:
        ff_est = estimate_fama_french(portfolio, market, benchmarks, rf)
        portfolio_betas = ff_est.iloc[0, 1:].to_numpy()
        betas.append(portfolio_betas)

    # 2nd step: estimate lambdas
    aligned_returns = align(
        *[adjust(portfolio.returns, rf) for portfolio in portfolios],
        *[adjust(market.returns, rf)] + list(benchmarks)
    )
    portfolios_returns = aligned_returns[:-len(benchmarks) - 1]

    x = np.array(betas)
    ys = np.array(portfolios_returns).T

    lambdas = []
    for y in ys:
        est = sm.OLS(y, x).fit()
        lambdas.append(est.params[1:])

    return pd.DataFrame(
        np.array([
            np.mean(lambdas, axis=0),
            np.std(lambdas, axis=0) / np.sqrt(len(lambdas))
        ]).T,
        index=[market.index.name] + [benchmark.index.name for benchmark in benchmarks],
        columns=["coef", "se"]
    )
