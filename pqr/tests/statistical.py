from __future__ import annotations

__all__ = [
    "t_test",
    "wilcoxon_test",
]

from typing import Literal

from scipy.stats import (
    ttest_1samp,
    wilcoxon,
)

from pqr.core import Portfolio


def t_test(
        portfolio: Portfolio,
        h0: float = 0,
        alternative: Literal["two-sided", "less", "greater"] = "greater",
) -> tuple[float, float]:
    t_stat, p_val = ttest_1samp(
        portfolio.returns,
        popmean=h0,
        alternative=alternative,
    )
    return t_stat, p_val


def wilcoxon_test(
        wml_portfolio: Portfolio,
        alternative: Literal["two-sided", "less", "greater"] = "greater",
) -> tuple[float, float]:
    t_stat, p_val = wilcoxon(
        wml_portfolio.returns,
        alternative=alternative,
    )
    return t_stat, p_val
