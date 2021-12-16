from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Optional, Protocol

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

from pqr.core import Portfolio, Benchmark
from .metrics import NumericMetric, TimeSeriesMetric

__all__ = [
    "Dashboard",
    "Chart",
    "Table",
]


class Displayable(Protocol):
    def display(self, portfolios: Sequence[Portfolio]) -> None:
        pass


@dataclass
class Dashboard:
    items: Sequence[Displayable]

    def display(self, portfolios: Sequence[Portfolio]) -> None:
        for item in self.items:
            item.display(portfolios)


@dataclass
class Chart:
    metric: NumericMetric | TimeSeriesMetric
    benchmark: Optional[Benchmark] = None
    log_scale: bool = False
    figsize: tuple[int, int] = (10, 10)

    def display(self, portfolios: Sequence[Portfolio]) -> None:
        plt.figure(figsize=self.figsize)

        if isinstance(self.metric, NumericMetric):
            metric = self.metric.trailing
            metric_name = f"Trailing {self.metric.name}"
        else:
            metric = self.metric.calculate
            metric_name = self.metric.name

        for portfolio in portfolios:
            plt.plot(metric(portfolio), label=portfolio.name)

        if self.benchmark is not None:
            starts_from = min(portfolio.returns.index[0] for portfolio in portfolios)
            benchmark_returns = self.benchmark.returns[starts_from:]
            benchmark_returns.iloc[0] = 0.0
            benchmark = Benchmark(benchmark_returns, name=self.benchmark.name)

            plt.plot(
                metric(benchmark),
                label=self.benchmark.name,
                color="gray",
                alpha=0.8)

        if self.log_scale:
            plt.yscale("symlog")

        plt.title(f"Portfolios {metric_name}")
        plt.xlabel("Date")
        plt.ylabel(f"{metric_name}{' (log scale)' if self.log_scale else ''}")
        plt.legend()
        plt.grid()

        plt.show()


@dataclass
class Table:
    metrics: Sequence[NumericMetric]

    def display(self, portfolios: Sequence[Portfolio]) -> None:
        metrics = {}
        for metric in self.metrics:
            metrics[metric.name] = [
                metric.fancy(portfolio) for portfolio in portfolios]

        metrics_table = pd.DataFrame(
            metrics,
            index=[portfolio.name for portfolio in portfolios]).T

        display(metrics_table)
