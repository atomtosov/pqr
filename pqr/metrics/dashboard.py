from __future__ import annotations

__all__ = [
    "FancyMetric",
    "show_table",
    "plot_chart",
    "Dashboard",
]

from dataclasses import dataclass
from typing import (
    Callable,
    Sequence,
    Optional,
)

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

from pqr.core import Portfolio, Benchmark


class FancyMetric:
    def __init__(
            self,
            metric_fn: Callable[
                [pd.DataFrame | Portfolio | Benchmark, ...],
                float | tuple[float, float, float] | pd.Series
            ],
            precision: int = 2,
            multiplier: float = 1.0,
            name: Optional[str] = None,
            **kwargs,
    ) -> None:
        self.metric_fn = metric_fn
        self.precision = precision
        self.multiplier = multiplier
        self.kwargs = kwargs

        self.name = name or metric_fn.__name__

    def __call__(self, portfolio: Portfolio) -> str | pd.Series:
        metric_value = self.metric_fn(portfolio, **self.kwargs)

        if isinstance(metric_value, tuple):  # statistics
            coef, t_stat, p_value = metric_value
            return "{coef}{stars} ({t_stat})".format(
                coef=format(coef * self.multiplier, f".{self.precision}f"),
                stars="*" * self._count_stars(p_value),
                t_stat=format(t_stat, f".{self.precision}f"),
            )
        elif isinstance(metric_value, pd.Series):  # time-series
            return (metric_value * self.multiplier).round(self.precision)
        else:
            return format(
                metric_value * self.multiplier,
                f".{self.precision}f"
            )

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


def show_table(
        portfolios: Sequence[Portfolio],
        metrics: Sequence[FancyMetric],
) -> None:
    metrics_table = {}
    for metric in metrics:
        metrics_table[metric.name] = [
            metric(portfolio)
            for portfolio in portfolios
        ]

    display(
        pd.DataFrame(
            metrics_table,
            index=[portfolio.name for portfolio in portfolios]
        ).T
    )


def plot_chart(
        portfolios: Sequence[Portfolio],
        metric: FancyMetric,
        benchmark: Optional[Benchmark] = None,
        log_scale: bool = False,
        **kwargs
) -> None:
    plt.figure(**kwargs)

    for portfolio in portfolios:
        plt.plot(metric(portfolio), label=portfolio.name)

    if benchmark is not None:
        plt.plot(
            metric(
                benchmark.starting_from(
                    min(portfolio.returns.index[0] for portfolio in portfolios)
                )
            ),
            label=benchmark.name,
            color="gray",
            alpha=0.8,
            linestyle="--",
        )

    if log_scale:
        plt.yscale("symlog")

    plt.title(f"Portfolios {metric.name}")
    plt.xlabel("Date")
    plt.ylabel(metric.name)
    plt.legend()
    plt.grid()

    plt.show()


@dataclass
class Dashboard:
    items: Sequence[Callable[[Sequence[Portfolio]], None]]

    def display(self, portfolios: Sequence[Portfolio]):
        for item in self.items:
            item(portfolios)
