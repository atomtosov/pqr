from dataclasses import dataclass

import numpy as np
import pandas as pd

from pqr.core.factor import Factor

__all__ = [
    "Quantiles",
    "Top",
    "Bottom",
    "TimeSeries",
]


@dataclass
class Quantiles:
    min_q: float = 0.0
    max_q: float = 1.0

    def __call__(self, factor: Factor) -> pd.DataFrame:
        factor_values = factor.values.to_numpy()
        q = factor.quantile([self.min_q, self.max_q]).to_numpy()

        return pd.DataFrame(
            (q[:, [0]] <= factor_values) & (factor_values <= q[:, [1]]),
            index=factor.values.index.copy(),
            columns=factor.values.columns.copy()
        )


@dataclass
class Top:
    n: int = 10

    def __call__(self, factor: Factor) -> pd.DataFrame:
        factor_values = factor.values.to_numpy()
        top = factor.top(self.n).to_numpy()

        return pd.DataFrame(
            factor_values >= top if factor.is_better_more() else factor_values <= top,
            index=factor.values.index.copy(),
            columns=factor.values.columns.copy()
        )


@dataclass
class Bottom:
    n: int = 10

    def __call__(self, factor: Factor) -> pd.DataFrame:
        factor_values = factor.values.to_numpy()
        bottom = factor.bottom(self.n).to_numpy()

        return pd.DataFrame(
            factor_values <= bottom if factor.is_better_more() else factor_values >= bottom,
            index=factor.values.index.copy(),
            columns=factor.values.columns.copy()
        )


@dataclass
class TimeSeries:
    min_threshold: float = -np.inf
    max_threshold: float = np.inf

    def __call__(self, factor: Factor) -> pd.DataFrame:
        factor_values = factor.values.to_numpy()

        return pd.DataFrame(
            (self.min_threshold <= factor_values) & (factor_values <= self.max_threshold),
            index=factor.values.index.copy(),
            columns=factor.values.columns.copy()
        )
