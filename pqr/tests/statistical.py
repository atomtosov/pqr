__all__ = [
    "t_test",
]

from typing import (
    Literal,
    Tuple,
)

import pandas as pd
from scipy.stats import (
    ttest_1samp,
    wilcoxon,
)


def t_test(
        portfolio: pd.DataFrame,
        h0: float = 0,
        alternative: Literal["two-sided", "less", "greater"] = "greater",
) -> Tuple[float, float]:
    t_stat, p_val = ttest_1samp(
        portfolio["returns"],
        popmean=h0,
        alternative=alternative,
    )
    return t_stat, p_val


def wilcoxon_test(
        portfolio: pd.DataFrame,
        alternative: Literal["two-sided", "less", "greater"] = "greater",
) -> Tuple[float, float]:
    t_stat, p_val = wilcoxon(
        portfolio["returns"],
        alternative=alternative,
    )
    return t_stat, p_val
