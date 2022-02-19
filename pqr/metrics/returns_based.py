from __future__ import annotations

__all__ = [
    "compounded_returns",
    "total_return",
    "cagr",
    "mean_return",
    "volatility",
    "downside_risk",
    "win_rate",
    "drawdown",
    "max_drawdown",
    "value_at_risk",
    "expected_tail_loss",
    "expected_tail_reward",
    "rachev_ratio",
    "calmar_ratio",
    "sharpe_ratio",
    "omega_ratio",
    "sortino_ratio",
    "trailing_total_return",
    "trailing_cagr",
    "trailing_mean_return",
    "trailing_volatility",
    "trailing_downside_risk",
    "trailing_win_rate",
    "trailing_max_drawdown",
    "trailing_value_at_risk",
    "trailing_expected_tail_loss",
    "trailing_expected_tail_reward",
    "trailing_rachev_ratio",
    "trailing_calmar_ratio",
    "trailing_sharpe_ratio",
    "trailing_omega_ratio",
    "trailing_sortino_ratio",
]

from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

from pqr.metrics._infer import infer
from pqr.utils import adjust


@infer(returns=True)
def compounded_returns(returns: pd.Series) -> pd.Series:
    return (1 + returns).cumprod() - 1


@infer(returns=True)
def total_return(returns: pd.Series) -> float:
    return compounded_returns(returns).iat[-1]


@infer(returns=True, window=True)
def trailing_total_return(
        returns: pd.Series,
        *,
        window: Optional[int] = None,
) -> pd.Series:
    return returns.rolling(window).apply(total_return).iloc[window:]


@infer(returns=True, annualizer=True)
def cagr(
        returns: pd.Series,
        *,
        annualizer: Optional[float] = None,
) -> float:
    tr = total_return(returns)
    n = len(returns) / annualizer

    return (1 + tr) ** (1 / n) - 1


@infer(returns=True, annualizer=True, window=True)
def trailing_cagr(
        returns: pd.Series,
        *,
        annualizer: Optional[float] = None,
        window: Optional[int] = None
) -> pd.Series:
    return returns.rolling(window).apply(
        total_return,
        kwargs={
            "annualizer": annualizer,
        },
    ).iloc[window:]


@infer(returns=True, annualizer=True)
def mean_return(
        returns: pd.Series,
        *,
        statistics: bool = False,
        annualizer: Optional[float] = None,
) -> float | tuple[float, float, float]:
    mr = returns.mean() * annualizer

    if statistics:
        mr = (mr, *ttest_1samp(returns, 0, alternative="greater"))

    return mr


@infer(returns=True, annualizer=True, window=True)
def trailing_mean_return(
        returns: pd.Series,
        *,
        annualizer: Optional[float] = None,
        window: Optional[int] = None
) -> pd.Series:
    return returns.rolling(window).mean().iloc[window:] * annualizer


@infer(returns=True, annualizer=True)
def volatility(
        returns: pd.Series,
        *,
        annualizer: Optional[float] = None,
) -> float:
    return returns.std(ddof=1) * np.sqrt(annualizer)


@infer(returns=True, annualizer=True, window=True)
def trailing_volatility(
        returns: pd.Series,
        *,
        annualizer: Optional[float] = None,
        window: Optional[int] = None,
) -> pd.Series:
    return returns.rolling(window).std(ddof=1).iloc[window:] * np.sqrt(annualizer)


@infer(returns=True, annualizer=True)
def downside_risk(
        returns: pd.Series,
        *,
        mar: float = 0.0,
        annualizer: Optional[float] = None,
) -> float:
    adjusted_returns = adjust(returns, mar)
    returns_under_mar = np.clip(
        adjusted_returns,
        a_min=-np.inf, a_max=0
    )

    return np.sqrt((returns_under_mar ** 2).mean()) * np.sqrt(annualizer)


@infer(returns=True, annualizer=True, window=True)
def trailing_downside_risk(
        returns: pd.Series,
        *,
        mar: float = 0.0,
        annualizer: Optional[float] = None,
        window: Optional[int] = None,
) -> pd.Series:
    return returns.rolling(window).apply(
        downside_risk,
        kwargs={
            "mar": mar,
            "annualizer": annualizer,
        },
    ).iloc[window:]


@infer(returns=True)
def win_rate(returns: pd.Series) -> float:
    return (returns > 0).mean()


@infer(returns=True, window=True)
def trailing_win_rate(
        returns: pd.Series,
        *,
        window: Optional[int] = None,
) -> pd.Series:
    return (returns > 0).rolling(window).mean().iloc[window:]


@infer(returns=True)
def drawdown(returns: pd.Series) -> pd.Series:
    equity = 1 + compounded_returns(returns)
    high_water_mark = equity.cummax()
    return equity / high_water_mark - 1


@infer(returns=True)
def max_drawdown(returns: pd.Series) -> float:
    return drawdown(returns).min()


@infer(returns=True, window=True)
def trailing_max_drawdown(
        returns: pd.Series,
        *,
        window: Optional[int] = None
) -> pd.Series:
    return returns.rolling(window).apply(max_drawdown).iloc[window:]


@infer(returns=True, annualizer=True)
def value_at_risk(
        returns: pd.Series,
        *,
        cutoff: float = 0.05,
        annualizer: Optional[float] = None,
) -> float:
    return returns.quantile(cutoff) * np.sqrt(annualizer)


@infer(returns=True, annualizer=True, window=True)
def trailing_value_at_risk(
        returns: pd.Series,
        *,
        cutoff: float = 0.05,
        annualizer: Optional[float] = None,
        window: Optional[int] = None,
) -> pd.Series:
    return returns.rolling(window).quantile(cutoff) * np.sqrt(annualizer)


@infer(returns=True, annualizer=True)
def expected_tail_loss(
        returns: pd.Series,
        *,
        cutoff: float = 0.05,
        annualizer: Optional[float] = None,
) -> float:
    under_cutoff = returns < returns.quantile(cutoff)
    return returns[under_cutoff].mean() * np.sqrt(annualizer)


@infer(returns=True, annualizer=True, window=True)
def trailing_expected_tail_loss(
        returns: pd.Series,
        *,
        cutoff: float = 0.05,
        annualizer: Optional[float] = None,
        window: Optional[int] = None,
) -> pd.Series:
    return returns.rolling(window).apply(
        expected_tail_loss,
        kwargs={
            "cutoff": cutoff,
            "annualizer": annualizer,
        },
    ).iloc[window:]


@infer(returns=True, annualizer=True)
def expected_tail_reward(
        returns: pd.Series,
        *,
        cutoff: float = 0.95,
        annualizer: Optional[float] = None,
) -> float:
    above_cutoff = returns > returns.quantile(cutoff)
    return returns[above_cutoff].mean() * np.sqrt(annualizer)


@infer(returns=True, annualizer=True, window=True)
def trailing_expected_tail_reward(
        returns: pd.Series,
        *,
        cutoff: float = 0.95,
        annualizer: Optional[float] = None,
        window: Optional[int] = None,
) -> pd.Series:
    return returns.rolling(window).apply(
        expected_tail_reward,
        kwargs={
            "cutoff": cutoff,
            "annualizer": annualizer,
        },
    ).iloc[window:]


@infer(returns=True)
def rachev_ratio(
        returns: pd.Series,
        *,
        reward_cutoff: float = 0.95,
        risk_cutoff: float = 0.05,
) -> float:
    etr = expected_tail_reward(returns, reward_cutoff, annualizer=1)
    etl = expected_tail_loss(returns, risk_cutoff, annualizer=1)
    return -(etr / etl)


@infer(returns=True, window=True)
def trailing_rachev_ratio(
        returns: pd.Series,
        *,
        reward_cutoff: float = 0.95,
        risk_cutoff: float = 0.05,
        window: Optional[int] = None,
) -> pd.Series:
    etr = trailing_expected_tail_reward(returns, reward_cutoff,
                                        annualizer=1, window=window)
    etl = trailing_expected_tail_loss(returns, risk_cutoff,
                                      annualizer=1, window=window)
    return -(etr / etl)


@infer(returns=True, annualizer=True)
def calmar_ratio(
        returns: pd.Series,
        *,
        annualizer: Optional[float] = None,
) -> float:
    cagr_ = cagr(returns, annualizer=annualizer)
    max_dd = max_drawdown(returns)
    return -(cagr_ / max_dd)


@infer(returns=True, annualizer=True, window=True)
def trailing_calmar_ratio(
        returns: pd.Series,
        *,
        annualizer: Optional[float] = None,
        window: Optional[int] = None,
) -> pd.Series:
    cagr_ = trailing_cagr(returns, annualizer=annualizer, window=window)
    max_dd = trailing_max_drawdown(returns, window=window)
    return -(cagr_ / max_dd)


@infer(returns=True, annualizer=True)
def sharpe_ratio(
        returns: pd.Series,
        *,
        rf: float = 0.0,
        annualizer: Optional[float] = None,
) -> float:
    adjusted_returns = adjust(returns, rf)
    mr = mean_return(adjusted_returns, statistics=False, annualizer=1)
    std = volatility(adjusted_returns, annualizer=1)
    return (mr / std) * np.sqrt(annualizer)


@infer(returns=True, annualizer=True, window=True)
def trailing_sharpe_ratio(
        returns: pd.Series,
        *,
        rf: float = 0.0,
        annualizer: Optional[float] = None,
        window: Optional[int] = None
) -> pd.Series:
    return returns.rolling(window).apply(
        sharpe_ratio,
        kwargs={
            "rf": rf,
            "annualizer": annualizer,
        },
    ).iloc[window:]


@infer(returns=True)
def omega_ratio(
        returns: pd.Series,
        *,
        rf: float = 0.0,
) -> float:
    adjusted_returns = adjust(returns, rf)
    above = adjusted_returns[adjusted_returns > 0].sum()
    under = adjusted_returns[adjusted_returns < 0].sum()
    return -(above / under)


@infer(returns=True, window=True)
def trailing_omega_ratio(
        returns: pd.Series,
        *,
        rf: float = 0.0,
        window: Optional[int] = None,
) -> pd.Series:
    return returns.rolling(window).apply(
        omega_ratio,
        kwargs={
            "rf": rf,
        },
    ).iloc[window:]


@infer(returns=True, annualizer=True)
def sortino_ratio(
        returns: pd.Series,
        *,
        rf: float = 0.0,
        annualizer: Optional[float] = None,
) -> float:
    adjusted_returns = adjust(returns, rf)
    mr = mean_return(adjusted_returns, statistics=False, annualizer=1)
    dr = downside_risk(adjusted_returns, rf=0, annualizer=1)
    return (mr / dr) * np.sqrt(annualizer)


@infer(returns=True, annualizer=True, window=True)
def trailing_sortino_ratio(
        returns: pd.Series,
        *,
        rf: float = 0.0,
        annualizer: Optional[float] = None,
        window: Optional[int] = None,
) -> pd.Series:
    adjusted_returns = adjust(returns, rf)
    mr = trailing_mean_return(adjusted_returns, statistics=False, annualizer=1, window=window)
    dr = trailing_downside_risk(adjusted_returns, rf=0, annualizer=1, window=window)
    return (mr / dr) * np.sqrt(annualizer)
