from __future__ import annotations

from dataclasses import make_dataclass
from warnings import warn

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

from pqr.utils import align


def extract_annualizer(df_or_series: pd.DataFrame | pd.Series) -> float:
    freq_alias = {
        "A": 1, "AS": 1, "BYS": 1, "BA": 1, "BAS": 1, "RE": 1,  # yearly
        "Q": 4, "QS": 4, "BQ": 4, "BQS": 4,  # quarterly
        "M": 12, "MS": 12, "BM": 12, "BMS": 12, "CBM": 12, "CBMS": 12,  # monthly
        "W": 52,  # weekly
        "B": 252, "C": 252, "D": 252,  # daily
    }

    if not isinstance(df_or_series.index, pd.DatetimeIndex):
        raise TypeError("df or series must have pd.DateTimeIndex to infer periodicity")

    idx = df_or_series.index
    inferred_freq = getattr(idx, "inferred_freq", None)
    annualizer = freq_alias.get(inferred_freq)

    if annualizer is None:
        warn("periodicity of df or series cannot be determined correctly, estimation is used")
        years_approx = (idx[-1] - idx[0]).days / 365.25
        annualizer = len(idx) / years_approx

    return annualizer


def adjust(
        returns: pd.Series,
        rf: float | pd.Series
) -> pd.Series:
    if isinstance(rf, pd.Series):
        returns, rf = align(returns, rf)

    return returns - rf


def stats_container_factory(metric_name: str) -> type:
    return make_dataclass(
        metric_name,
        [
            ("value", float),
            ("t_stat", float),
            ("p_value", float),
        ],
        namespace={
            "template": property(lambda self: "{value:.2f}{stars} ({t_stat:.2f})"),
            "count_stars": lambda self: 3 if self.p_value < 0.01 else (
                2 if self.p_value < 0.05 else (
                    1 if self.p_value < 0.1 else 0
                )
            )
        }
    )


def estimate_ols(
        returns: pd.Series,
        benchmark: pd.Series,
        rf: float = 0.0
):
    adjusted_returns = adjust(returns, rf)
    adjusted_benchmark = adjust(benchmark, rf)

    y, x = align(adjusted_returns, adjusted_benchmark)
    x = sm.add_constant(x.to_numpy())
    ols = sm.OLS(y.to_numpy(), x)

    return ols.fit()


def estimate_rolling_ols(
        returns: pd.Series,
        benchmark: pd.Series,
        window: int,
        rf: float = 0.0,
):
    adjusted_returns = adjust(returns, rf)
    adjusted_benchmark = adjust(benchmark, rf)

    y, x = align(adjusted_returns, adjusted_benchmark)
    x = sm.add_constant(x.to_numpy())
    ols = RollingOLS(y.to_numpy(), x, window=window)

    return ols.fit()


def estimate_holding(picks: pd.DataFrame) -> int:
    diff = np.diff(picks.to_numpy(), axis=0)
    rebalancings_long = (diff == 1).any(axis=1).sum()
    rebalancings_short = (diff == -1).any(axis=1).sum()
    avg_rebalacings = (rebalancings_long + rebalancings_short) / 2

    return round(len(diff) / avg_rebalacings)
