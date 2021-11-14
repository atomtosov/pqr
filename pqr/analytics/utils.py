from __future__ import annotations

from dataclasses import make_dataclass
from typing import Callable, Optional
from warnings import warn

import pandas as pd
import statsmodels.api as sm

from pqr.utils import align

FREQ_ALIAS = {
    "A": 1, "AS": 1, "BYS": 1, "BA": 1, "BAS": 1, "RE": 1,  # yearly
    "Q": 4, "QS": 4, "BQ": 4, "BQS": 4,  # quarterly
    "M": 12, "MS": 12, "BM": 12, "BMS": 12, "CBM": 12, "CBMS": 12,  # monthly
    "W": 52,  # weekly
    "B": 252, "C": 252, "D": 252,  # daily
}


def extract_annualizer(df_or_series: pd.DataFrame | pd.Series) -> float:
    if not isinstance(df_or_series.index, pd.DatetimeIndex):
        raise TypeError("df or series must have pd.DateTimeIndex to infer periodicity")

    idx = df_or_series.index
    inferred_freq = getattr(idx, "inferred_freq", None)
    annualizer = FREQ_ALIAS.get(inferred_freq)

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
                2 if self.p_value < 0.05 else(
                    1 if self.p_value < 0.1 else 0
                )
            )
        }
    )


def stat_significance(p_value: float) -> str:
    if p_value < 0.01:
        stars = 3
    elif p_value < 0.05:
        stars = 2
    elif p_value < 0.1:
        stars = 1
    else:
        stars = 0

    return "*" * stars


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


def trail(
        df_or_series: pd.DataFrame | pd.Series,
        func: Callable[[pd.Series | pd.DataFrame], float],
        window: Optional[int] = None,
) -> pd.Series:
    if window is None:
        window = round(extract_annualizer(df_or_series))

    values = []
    for i in range(window, len(df_or_series[0]) + 1):
        values.append(
            func(df_or_series.iloc[i - window:i])
        )

    return pd.Series(
        values,
        index=df_or_series.index[window:].copy()
    )


class Stats:
    value: float
    t_stat: float
    p_value: float

    metric_name: str

    def fancy(self) -> str:
        return "{value:.2f}{stars} {t_stat:.2f}".format(
            value=self.value,
            stars=self.stars,
            t_stat=self.t_stat
        )

    @property
    def stars(self) -> str:
        if self.p_value < 0.01:
            stars = 3
        elif self.p_value < 0.05:
            stars = 2
        elif self.p_value < 0.1:
            stars = 1
        else:
            stars = 0

        return "*" * stars
