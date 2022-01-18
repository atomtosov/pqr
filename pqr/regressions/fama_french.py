__all__ = [
    "estimate_fama_french",
]

from typing import (
    Optional,
    Sequence,
)

import numpy as np
import pandas as pd
import statsmodels.api as sm

from pqr.utils import (
    adjust,
    estimate_annualizer,
    align,
)


def estimate_fama_french(
        portfolio: pd.DataFrame,
        market: pd.Series,
        benchmarks: Sequence[pd.Series],
        rf: float = 0.0,
        annualizer: Optional[float] = None,
) -> pd.DataFrame:
    portfolio_returns, market_returns, *benchmarks_returns = align(
        adjust(portfolio["returns"], rf),
        adjust(market, rf),
        *[benchmark for benchmark in benchmarks]
    )

    y = portfolio_returns.to_numpy()
    x = np.hstack([
        market_returns.to_numpy()[:, np.newaxis],
        benchmarks_returns.to_numpy()
    ])
    x = sm.add_constant(x)
    est = sm.OLS(y, x).fit()

    idx = (["alpha", f"beta_{market.index.name}"] +
           [f"beta_{benchmark.index.name}" for benchmark in benchmarks])
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
