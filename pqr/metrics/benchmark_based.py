from __future__ import annotations

__all__ = [
    "benchmark_correlation",
    "excess_returns",
    "mean_excess_return",
    "alpha",
    "beta",
    "trailing_benchmark_correlation",
    "trailing_mean_excess_return",
    "trailing_alpha",
    "trailing_beta",
]

from typing import Optional

import pandas as pd
import statsmodels.api as sm
from scipy.stats import ttest_1samp
from statsmodels.regression.linear_model import RegressionResults
from statsmodels.regression.rolling import (
    RollingRegressionResults,
    RollingOLS,
)

from pqr.metrics._infer import infer
from pqr.utils import (
    adjust,
    align,
)


@infer(returns=True, benchmark=True)
def benchmark_correlation(
        returns: pd.Series,
        benchmark: pd.Series,
) -> float:
    returns, benchmark = align(returns, benchmark)
    return returns.corr(benchmark)


@infer(returns=True, benchmark=True)
def trailing_benchmark_correlation(
        returns: pd.Series,
        benchmark: pd.Series,
        *,
        window: Optional[int] = None
) -> pd.Series:
    returns, benchmark = align(returns, benchmark)
    return returns.rolling(window).corr(benchmark).iloc[window:]


@infer(returns=True, benchmark=True)
def excess_returns(
        returns: pd.Series,
        benchmark: pd.Series,
) -> pd.Series:
    return adjust(returns, benchmark)


@infer(returns=True, benchmark=True, annualizer=True)
def mean_excess_return(
        returns: pd.Series,
        benchmark: pd.Series,
        *,
        statistics: bool = False,
        annualizer: Optional[float] = None,
) -> float | tuple[float, float, float]:
    er = excess_returns(returns, benchmark)
    mer = er.mean() * annualizer

    if statistics:
        t_stat, p_value = ttest_1samp(er, 0, alternative="greater")
        mer = (mer, t_stat, p_value)

    return mer


@infer(returns=True, benchmark=True, annualizer=True, window=True)
def trailing_mean_excess_return(
        returns: pd.Series,
        benchmark: pd.Series,
        *,
        annualizer: Optional[float] = None,
        window: Optional[int] = None,
) -> pd.Series:
    er = excess_returns(returns, benchmark)
    return er.rolling(window).mean().iloc[window:] * annualizer


@infer(returns=True, benchmark=True, annualizer=True)
def alpha(
        returns: pd.Series,
        benchmark: pd.Series,
        *,
        rf: float = 0.0,
        statistics: bool = False,
        annualizer: Optional[float] = None,
) -> float | tuple[float, float, float]:
    capm = _estimate_capm(returns, benchmark, rf)
    alpha_capm = capm.params[0] * annualizer

    if statistics:
        alpha_capm = (alpha_capm, capm.tvalues[0], capm.pvalues[0])

    return alpha_capm


@infer(returns=True, benchmark=True, annualizer=True, window=True)
def trailing_alpha(
        returns: pd.Series,
        benchmark: pd.Series,
        *,
        rf: float = 0.0,
        annualizer: Optional[float] = None,
        window: Optional[int] = None,
) -> pd.Series:
    capm = _estimate_trailing_capm(returns, benchmark, rf, window)
    return pd.Series(
        capm.params[window:, 0] * annualizer,
        index=returns.index[-len(capm.params) + window:]
    )


@infer(returns=True, benchmark=True)
def beta(
        returns: pd.Series,
        benchmark: pd.Series,
        *,
        rf: float = 0.0,
        statistics: bool = False,
) -> float | tuple[float, float, float]:
    capm = _estimate_capm(returns, benchmark, rf)
    beta_capm = capm.params[1]

    if statistics:
        beta_capm = (beta_capm, capm.tvalues[1], capm.pvalues[1])

    return beta_capm


@infer(returns=True, benchmark=True, window=True)
def trailing_beta(
        returns: pd.Series,
        benchmark: pd.Series,
        *,
        rf: float = 0.0,
        window: Optional[int] = None,
) -> pd.Series:
    capm = _estimate_trailing_capm(returns, benchmark, rf, window)
    return pd.Series(
        capm.params[window:, 1],
        index=returns.index[-len(capm.params) + window:]
    )


def _estimate_capm(
        returns: pd.Series,
        benchmark: pd.Series,
        rf: float,
) -> RegressionResults:
    returns, benchmark = align(
        adjust(returns, rf),
        adjust(benchmark, rf)
    )

    y = returns.to_numpy()
    x = sm.add_constant(benchmark.to_numpy())
    return sm.OLS(y, x).fit()


def _estimate_trailing_capm(
        returns: pd.Series,
        benchmark: pd.Series,
        rf: float,
        window: int,
) -> RollingRegressionResults:
    returns, benchmark = align(
        adjust(returns, rf),
        adjust(benchmark, rf)
    )

    y = returns.to_numpy()
    x = sm.add_constant(benchmark.to_numpy())
    return RollingOLS(y, x, window=window).fit()
