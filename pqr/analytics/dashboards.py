from __future__ import annotations

from typing import Sequence, Protocol, Optional

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

from pqr.core import Portfolio, Benchmark

__all__ = [
    "Dashboard",
    "Graph",
    "Table",
]


class NumericMetric(Protocol):
    def fancy(self, portfolio: Portfolio) -> str:
        pass

    @property
    def fancy_name(self, portfolio: Portfolio) -> str:
        pass


class TimeSeriesMetric(Protocol):
    def __call__(self, portfolio: Portfolio) -> pd.Series:
        pass

    def fancy(self, portfolio: Portfolio) -> pd.Series:
        pass

    @property
    def fancy_name(self, portfolio: Portfolio) -> str:
        pass


class Dashboard:
    def __init__(self, *items: Graph | Table):
        self.items = items

    def __call__(self, portfolios: Sequence[Portfolio]) -> None:
        for item in self.items:
            item(portfolios)


class Graph:
    def __init__(
            self,
            metric: TimeSeriesMetric,
            benchmark: Optional[Benchmark] = None,
            log_scale: bool = False,
            figsize: tuple[int, int] = (10, 10)
    ):
        self.metric = metric
        self.benchmark = benchmark
        self.log_scale = log_scale
        self.figsize = figsize

    def __call__(self, portfolios: Sequence[Portfolio]) -> None:
        plt.figure(figsize=self.figsize)

        for portfolio in portfolios:
            plt.plot(
                self.metric(portfolio) if self.log_scale else
                self.metric.fancy(portfolio),
                label=portfolio.name
            )

        if self.benchmark is not None:
            starts_from = min(portfolio.returns.index[0] for portfolio in portfolios)
            benchmark_returns = self.benchmark.returns[starts_from:]
            benchmark_returns.iloc[0] = 0.0
            benchmark = Benchmark(
                benchmark_returns,
                name=self.benchmark.name
            )
            plt.plot(
                self.metric(benchmark) if self.log_scale else
                self.metric.fancy(benchmark),
                label=self.benchmark.name,
                color="gray",
                alpha=0.8,
            )

        if self.log_scale:
            plt.yscale("symlog")

        plt.title(f"Portfolios {self.metric.fancy_name}")
        plt.xlabel("Date")
        plt.ylabel(f"{self.metric.fancy_name}{' (log scale)' if self.log_scale else ''}")
        plt.legend()
        plt.grid()

        plt.show()


class Table:
    def __init__(self, *metrics: NumericMetric):
        self.metrics = metrics

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
