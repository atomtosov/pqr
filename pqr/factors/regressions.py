__all__ = [
    "estimate_fama_french",
    "estimate_fama_macbeth",
]

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm

from pqr.core import Portfolio, Benchmark
from pqr.utils import (
    adjust,
    estimate_annualizer,
    align,
)


def estimate_fama_french(
        portfolio: Portfolio,
        market: Benchmark,
        benchmarks: Sequence[Benchmark],
        rf: float = 0.0,
        annualizer: Optional[float] = None,
) -> pd.DataFrame:
    portfolio_returns, market_returns, *benchmarks_returns = align(
        adjust(portfolio.returns, rf),
        adjust(market.returns, rf),
        *[benchmark.returns for benchmark in benchmarks]
    )

    y = portfolio_returns.to_numpy()
    x = np.hstack([
        market_returns.to_numpy()[:, np.newaxis],
        benchmarks_returns.to_numpy()
    ])
    x = sm.add_constant(x)
    est = sm.OLS(y, x).fit()

    idx = (["alpha", f"beta_{market.name}"] +
           [f"beta_{benchmark.name}" for benchmark in benchmarks])
    cols = ["coef", "se", "t_stat", "p_value"]
    rows = []
    for i in range(len(est.params)):
        rows.append([
            est.params[i],
            est.bse[i],
            est.tvalues[i],
            est.pvalues[i]
        ])
    table = pd.DataFrame(rows, index=idx, columns=cols)

    # annualize alpha and turn it to pct
    if annualizer is None:
        annualizer = estimate_annualizer(portfolio_returns)
    table.iloc[0, 0] *= annualizer * 100

    return table


def estimate_fama_macbeth(
        portfolios: Sequence[Portfolio],
        market: Benchmark,
        benchmarks: Sequence[Benchmark],
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
        adjust(market.returns, rf),
        *[benchmark.returns for benchmark in benchmarks]
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
        index=[market.name] + [benchmark.name for benchmark in benchmarks],
        columns=["coef", "se"]
    )
