from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Callable

import pandas as pd

from pqr.core.factor import Factorizer, LookBack, Lag, Hold
from pqr.core.portfolio import Portfolio
from .factor_model import FactorModel

__all__ = [
    "GridSearch",
]

AggregationFunction = Callable[[pd.DataFrame], pd.Series]
Metric = Callable[[Portfolio], float]


@dataclass
class GridSearch:
    params: list[tuple[int, int, int]]
    factor_model: FactorModel

    def __call__(
            self,
            factor_values: pd.DataFrame,
            agg_func: AggregationFunction,
            better: Literal["more", "less"],
            target: Metric,
    ) -> pd.DataFrame:
        metrics = []

        for looking, lag, holding in self.params:
            factorizer = Factorizer(
                [
                    LookBack(agg_func, looking),
                    Lag(lag),
                    Hold(holding)
                ]
            )
            factor = factorizer(factor_values, better)

            portfolios = self.factor_model(factor)

            metrics.append(
                pd.DataFrame(
                    [[target(portfolio) for portfolio in portfolios]],
                    index=[(looking, lag, holding)],
                    columns=[portfolio.name for portfolio in portfolios]
                )
            )

        return pd.concat(metrics)
