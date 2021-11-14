from dataclasses import dataclass, field
from typing import Protocol, Sequence, Optional

import matplotlib.pyplot as plt
import pandas as pd

from pqr.core import Portfolio, Benchmark


class TimeSeriesMetric(Protocol):
    def fancy(self, portfolio: Portfolio) -> pd.Series:
        pass

    @property
    def fancy_name(self, portfolio: Portfolio) -> str:
        pass


@dataclass
class Graph:
    metric: TimeSeriesMetric
    benchmark: Optional[Benchmark] = None
    log_scale: bool = False
    hlines: Sequence[float] = field(default_factory=list)

    def __call__(self, portfolios: Sequence[Portfolio]) -> None:
        for portfolio in portfolios:
            plt.plot(
                self.metric.fancy(portfolio),
                label=portfolio.name
            )

        if self.benchmark is not None:
            plt.plot(
                self.metric.fancy(self.benchmark),
                label=self.benchmark.name,
                color="gray",
                alpha=0.75,
            )

        if self.log_scale:
            plt.yscale("symlog")

        plt.title(f"Portfolios {self.metric.fancy_name}")
        plt.xlabel("Date")
        plt.ylabel(self.metric.fancy_name)
        plt.legend()
        plt.grid()

        plt.show()
