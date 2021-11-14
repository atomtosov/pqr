from dataclasses import dataclass
from typing import Protocol, Sequence

import pandas as pd
from IPython.display import display

from pqr.core import Portfolio


class NumericMetric(Protocol):
    def fancy(self, portfolio: Portfolio) -> str:
        pass

    @property
    def fancy_name(self, portfolio: Portfolio) -> str:
        pass


@dataclass
class Table:
    metrics: Sequence[NumericMetric]

    def __call__(self, portfolios: Sequence[Portfolio]) -> None:
        metrics_table = {}
        for metric in self.metrics:
            metrics_table[metric.fancy_name] = [
                metric.fancy(portfolio) for portfolio in portfolios
            ]

        display(
            pd.DataFrame(
                metrics_table,
                index=[portfolio.name for portfolio in portfolios]
            ).T
        )
