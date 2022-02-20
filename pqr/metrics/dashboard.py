from __future__ import annotations

__all__ = [
    "Table",
    "Figure",
    "Dashboard",
]

from dataclasses import dataclass, field
from functools import wraps
from typing import (
    Callable,
    Optional,
)

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

from pqr.core import Portfolio, Benchmark


@dataclass
class Table:
    metrics: dict = field(default_factory=dict)

    def display(self, portfolios: list[Portfolio]) -> None:
        metrics_table = {}
        for name, metric in self.metrics.items():
            metrics_table[name] = [
                metric(portfolio)
                for portfolio in portfolios
            ]

        display(
            pd.DataFrame(
                metrics_table,
                index=[portfolio.name for portfolio in portfolios]
            ).T
        )

    def add_metric(
            self,
            metric: Callable[[Portfolio], float | tuple[float, float, float]],
            *,
            multiplier: float = 1.0,
            precision: int = 2,
            name: Optional[str] = None,
    ) -> None:
        @wraps(metric)
        def displayable_metric(portfolio: Portfolio) -> str:
            metric_value = metric(portfolio)

            if isinstance(metric_value, tuple):  # statistics
                coef, t_stat, p_value = metric_value
                return "{coef}{stars} ({t_stat})".format(
                    coef=format(coef * multiplier, f".{precision}f"),
                    stars="*" * self._count_stars(p_value),
                    t_stat=format(t_stat, f".{precision}f"),
                )
            else:
                return format(
                    metric_value * multiplier,
                    f".{precision}f"
                )

        if name is None:
            name = metric.__name__
        self.metrics[name] = displayable_metric

    @staticmethod
    def _count_stars(p_value: float) -> int:
        if p_value < 0.01:
            return 3
        elif p_value < 0.05:
            return 2
        elif p_value < 0.1:
            return 1
        else:
            return 0


@dataclass
class Figure:
    metric: Callable[[Portfolio | Benchmark], pd.Series]
    multiplier: float = 1.0
    name: Optional[str] = None
    benchmark: Optional[Benchmark] = None
    log_scale: bool = False
    kwargs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.name is None:
            self.name = self.metric.__name__

    def display(self, portfolios: list[Portfolio]) -> None:
        plt.figure(**self.kwargs)

        for portfolio in portfolios:
            plt.plot(self.metric(portfolio) * self.multiplier, label=portfolio.name)

        if self.benchmark is not None:
            plt.plot(
                self.metric(
                    self.benchmark.starting_from(
                        min(portfolio.returns.index[0] for portfolio in portfolios)
                    )
                ) * self.multiplier,
                label=self.benchmark.name,
                color="gray",
                alpha=0.8,
                linestyle="--",
            )

        if self.log_scale:
            plt.yscale("symlog")

        plt.title(f"Portfolios {self.name}")
        plt.xlabel("Date")
        plt.ylabel(self.name)
        plt.legend()
        plt.grid()

        plt.show()


@dataclass
class Dashboard:
    items: list[Table | Figure] = field(default_factory=list)

    def display(self, portfolios: list[Portfolio]):
        for item in self.items:
            item.display(portfolios)

    def add_item(self, item: Table | Figure) -> None:
        self.items.append(item)

    def remove_item(self, idx: int) -> None:
        self.items.pop(idx)
