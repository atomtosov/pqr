from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Protocol, Optional

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

from pqr.core import Portfolio, Benchmark
from .metrics import *

__all__ = [
    "Dashboard",
    "Graph",
    "Table",

    "SummaryDashboard",
]


class NumericMetric(Protocol):
    def fancy(self, portfolio: Portfolio) -> str:
        pass

    @property
    def fancy_name(self, portfolio: Portfolio) -> str:
        pass


class TimeSeriesMetric(Protocol):
    def fancy(self, portfolio: Portfolio) -> pd.Series:
        pass

    @property
    def fancy_name(self, portfolio: Portfolio) -> str:
        pass


class Dashboard:
    def __init__(self, items: Sequence[Graph | Table]):
        self.items = items

    def __call__(self, portfolios: Sequence[Portfolio]) -> None:
        for item in self.items:
            item(portfolios)


class Graph:
    def __init__(
            self,
            metric: TimeSeriesMetric,
            benchmark: Optional[Benchmark],
            log_scale: bool = False
    ):
        self.metric = metric
        self.benchmark = benchmark
        self.log_scale = log_scale

    def __call__(self, portfolios: Sequence[Portfolio]) -> None:
        for portfolio in portfolios:
            plt.plot(
                self.metric.fancy(portfolio),
                label=portfolio.name
            )

        if self.benchmark is not None:
            starts_from = min(portfolio.returns.index[0] for portfolio in portfolios)
            benchmark = Benchmark(
                self.benchmark.returns[starts_from:],
                name=self.benchmark.name
            )
            plt.plot(
                self.metric.fancy(benchmark),
                label=self.benchmark.name,
                color="gray",
                alpha=0.8,
            )

        if self.log_scale:
            plt.yscale("symlog")

        plt.title(f"Portfolios {self.metric.fancy_name}")
        plt.xlabel("Date")
        plt.ylabel(self.metric.fancy_name)
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


@dataclass
class SummaryDashboard:
    benchmark: Benchmark
    log_scale: bool = False
    rf: float = 0.0
    annualizer: Optional[float] = None

    def __post_init__(self):
        self._dashboard = Dashboard(
            [
                Graph(
                    CompoundedReturns(),
                    benchmark=self.benchmark,
                    log_scale=self.log_scale,
                ),

                Table(
                    TotalReturn(),
                    CAGR(annualizer=self.annualizer),
                    MeanReturn(statistics=True, annualizer=self.annualizer),
                    Volatility(annualizer=self.annualizer),
                    WinRate(),
                    MeanTurnover(annualizer=self.annualizer),
                    MaxDrawdown(),
                    ValueAtRisk(annualizer=self.annualizer),
                    ExpectedTailLoss(annualizer=self.annualizer),
                    ExpectedTailReward(annualizer=self.annualizer),
                    RachevRatio(),
                    CalmarRatio(),
                    SharpeRatio(rf=self.rf, annualizer=self.annualizer),
                    OmegaRatio(),
                    SortinoRatio(),
                    MeanExcessReturn(statistics=True, benchmark=self.benchmark, annualizer=self.annualizer),
                    Alpha(statistics=True, benchmark=self.benchmark, rf=self.rf, annualizer=self.annualizer),
                    Beta(statistics=True, benchmark=self.benchmark, rf=self.rf),
                )
            ]
        )

    def __call__(self, portfolios: Sequence[Portfolio]) -> None:
        self._dashboard(portfolios)
