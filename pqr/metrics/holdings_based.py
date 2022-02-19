from __future__ import annotations

__all__ = [
    "turnover",
    "mean_turnover",
    "trailing_mean_turnover",
]

from typing import Optional

import numpy as np
import pandas as pd

from pqr.metrics._infer import infer


@infer(holdings=True)
def turnover(holdings: pd.DataFrame) -> pd.Series:
    holdings_array = holdings.to_numpy()
    abs_holdings_change = np.abs(np.diff(holdings_array, axis=0))

    return pd.Series(
        np.insert(
            np.nansum(abs_holdings_change, axis=1),
            obj=0, values=np.nansum(np.abs(holdings_array[0]))
        ),
        index=holdings.index.copy(),
    )


@infer(holdings=True, annualizer=True)
def mean_turnover(
        holdings: pd.DataFrame,
        *,
        annualizer: Optional[float] = None,
) -> float:
    return turnover(holdings).mean() * annualizer


@infer(holdings=True, annualizer=True, window=True)
def trailing_mean_turnover(
        holdings: pd.DataFrame,
        *,
        annualizer: Optional[float] = None,
        window: Optional[int] = None,
) -> pd.Series:
    return turnover(holdings).rolling(window).mean().iloc[window:] * annualizer
