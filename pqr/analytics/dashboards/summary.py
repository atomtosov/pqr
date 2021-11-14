from dataclasses import dataclass
from typing import Sequence, Optional

from pqr.analytics.metrics.numeric import *
from pqr.analytics.metrics.timeseries import CompoundedReturns
from pqr.core import Portfolio, Benchmark
from .dashboard import Dashboard
from .graph import Graph
from .table import Table

__all__ = [
    "SummaryDashboard",
]


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
                    log_scale=self.log_scale
                ),

                Table(
                    [
                        TotalReturn(),
                        CAGR(annualizer=self.annualizer),
                        MeanReturn(statistics=True, annualizer=self.annualizer),
                        Volatility(annualizer=self.annualizer),
                        WinRate(),
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
                    ]
                )
            ]
        )

    def __call__(self, portfolios: Sequence[Portfolio]) -> None:
        self._dashboard(portfolios)
