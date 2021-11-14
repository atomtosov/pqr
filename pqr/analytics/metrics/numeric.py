from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

from pqr.analytics.utils import extract_annualizer, adjust, estimate_ols, stats_container_factory
from pqr.core import Portfolio, Benchmark
from pqr.utils import align
from .timeseries import CompoundedReturns, Drawdown, Turnover

__all__ = [
    "TotalReturn",
    "CAGR",
    "MeanReturn",
    "Volatility",
    "WinRate",
    "MaxDrawdown",
    "ValueAtRisk",
    "ExpectedTailLoss",
    "ExpectedTailReward",
    "RachevRatio",
    "CalmarRatio",
    "SharpeRatio",
    "OmegaRatio",
    "SortinoRatio",
    "BenchmarkCorrelation",
    "MeanExcessReturn",
    "Alpha",
    "Beta"
]


class Stats(Protocol):
    value: float
    t_stat: float
    p_value: float

    def count_stars(self) -> int:
        pass

    @property
    def template(self) -> str:
        pass


@dataclass
class TotalReturn:
    def __call__(self, portfolio: Portfolio) -> pd.Series:
        return CompoundedReturns()(portfolio).iloc[-1]

    def fancy(self, portfolio: Portfolio) -> str:
        return format(self(portfolio) * 100, ".2f")

    @property
    def fancy_name(self) -> str:
        return "Total Return, %"


@dataclass
class CAGR:
    annualizer: Optional[float] = None

    def __call__(self, portfolio: Portfolio) -> float:
        if self.annualizer is None:
            annualizer = extract_annualizer(portfolio.returns)
        else:
            annualizer = self.annualizer

        tr = TotalReturn()(portfolio)
        years = len(portfolio.returns) / annualizer

        return (1 + tr) ** (1 / years) - 1

    def fancy(self, portfolio: Portfolio) -> str:
        return format(self(portfolio) * 100, ".2f")

    @property
    def fancy_name(self) -> str:
        return "CAGR, %"


@dataclass
class MeanReturn:
    statistics: bool = False
    annualizer: Optional[float] = None

    def __call__(self, portfolio: Portfolio) -> float | Stats:
        if self.annualizer is None:
            annualizer = extract_annualizer(portfolio.returns)
        else:
            annualizer = self.annualizer

        mr = portfolio.returns.mean() * annualizer

        if self.statistics:
            ttest = ttest_1samp(portfolio.returns, 0, alternative="greater")
            mr = stats_container_factory("MeanReturn")(
                value=mr,
                t_stat=ttest.statistic,
                p_value=ttest.pvalue
            )

        return mr

    def fancy(self, portfolio: Portfolio) -> str:
        mr = self(portfolio)
        if self.statistics:
            return mr.template.format(
                value=mr.value * 100,
                stars="*" * mr.count_stars(),
                t_stat=mr.t_stat
            )

        return format(mr * 100, ".2f")

    @property
    def fancy_name(self) -> str:
        return "Mean Return, %"


@dataclass
class Volatility:
    annualizer: Optional[float] = None

    def __call__(self, portfolio: Portfolio) -> float:
        if self.annualizer is None:
            annualizer = extract_annualizer(portfolio.returns)
        else:
            annualizer = self.annualizer

        return portfolio.returns.std() * np.sqrt(annualizer)

    def fancy(self, portfolio: Portfolio) -> str:
        return format(self(portfolio) * 100, ".2f")

    @property
    def fancy_name(self) -> str:
        return "Volatility, %"


@dataclass
class WinRate:
    def __call__(self, portfolio: Portfolio) -> float:
        return (portfolio.returns > 0).sum() / len(portfolio.returns)

    def fancy(self, portfolio: Portfolio) -> str:
        return format(self(portfolio) * 100, ".2f")

    @property
    def fancy_name(self) -> str:
        return "Win Rate, %"


@dataclass
class MaxDrawdown:
    def __call__(self, portfolio: Portfolio) -> float:
        return Drawdown()(portfolio).min()

    def fancy(self, portfolio: Portfolio) -> str:
        return format(self(portfolio) * 100, ".2f")

    @property
    def fancy_name(self) -> str:
        return "Maximum Drawdown, %"


@dataclass
class ValueAtRisk:
    cutoff: float = 0.05
    annualizer: Optional[float] = None

    def __call__(self, portfolio: Portfolio) -> float:
        if self.annualizer is None:
            annualizer = extract_annualizer(portfolio.returns)
        else:
            annualizer = self.annualizer

        return portfolio.returns.quantile(self.cutoff) * np.sqrt(annualizer)

    def fancy(self, portfolio: Portfolio) -> str:
        return format(self(portfolio) * 100, ".2f")

    @property
    def fancy_name(self) -> str:
        return "Value at Risk, %"


@dataclass
class ExpectedTailLoss:
    cutoff: float = 0.05
    annualizer: Optional[float] = None

    def __call__(self, portfolio: Portfolio) -> float:
        if self.annualizer is None:
            annualizer = extract_annualizer(portfolio.returns)
        else:
            annualizer = self.annualizer

        less_cutoff = portfolio.returns <= portfolio.returns.quantile(self.cutoff)
        return portfolio.returns[less_cutoff].mean() * np.sqrt(annualizer)

    def fancy(self, portfolio: Portfolio) -> str:
        return format(self(portfolio) * 100, ".2f")

    @property
    def fancy_name(self) -> str:
        return "Expected Tail Loss, %"


@dataclass
class ExpectedTailReward:
    cutoff: float = 0.95
    annualizer: Optional[float] = None

    def __call__(self, portfolio: Portfolio) -> float:
        if self.annualizer is None:
            annualizer = extract_annualizer(portfolio.returns)
        else:
            annualizer = self.annualizer

        more_cutoff = portfolio.returns >= portfolio.returns.quantile(self.cutoff)
        return portfolio.returns[more_cutoff].mean() * np.sqrt(annualizer)

    def fancy(self, portfolio: Portfolio) -> str:
        return format(self(portfolio) * 100, ".2f")

    @property
    def fancy_name(self) -> str:
        return "Expected Tail Reward, %"


@dataclass
class RachevRatio:
    reward_cutoff: float = 0.95
    risk_cutoff: float = 0.05

    def __call__(self, portfolio: Portfolio) -> float:
        etr = ExpectedTailReward(self.reward_cutoff)(portfolio)
        etl = ExpectedTailLoss(self.risk_cutoff)(portfolio)
        return -(etr / etl)

    def fancy(self, portfolio: Portfolio) -> str:
        return format(self(portfolio), ".2f")

    @property
    def fancy_name(self) -> str:
        return "Rachev Ratio"


@dataclass
class CalmarRatio:
    annualizer: Optional[float] = None

    def __call__(self, portfolio: Portfolio) -> float:
        if self.annualizer is None:
            annualizer = extract_annualizer(portfolio.returns)
        else:
            annualizer = self.annualizer

        return -(CAGR(annualizer)(portfolio) / MaxDrawdown()(portfolio))

    def fancy(self, portfolio: Portfolio) -> str:
        return format(self(portfolio), ".2f")

    @property
    def fancy_name(self) -> str:
        return "Calmar Ratio"


@dataclass
class SharpeRatio:
    rf: float = 0.0
    annualizer: Optional[float] = None

    def __call__(self, portfolio: Portfolio) -> float:
        adjusted = adjust(portfolio.returns, self.rf)
        if self.annualizer is None:
            annualizer = extract_annualizer(adjusted)
        else:
            annualizer = self.annualizer

        return adjusted.mean() / adjusted.std() * np.sqrt(annualizer)

    def fancy(self, portfolio: Portfolio) -> str:
        return format(self(portfolio), ".2f")

    @property
    def fancy_name(self) -> str:
        return "Sharpe Ratio"


@dataclass
class OmegaRatio:
    rf: float = 0.0

    def __call__(self, portfolio: Portfolio) -> float:
        adjusted = adjust(portfolio.returns, self.rf)
        above = adjusted[adjusted > 0].sum()
        under = adjusted[adjusted < 0].sum()
        return -(above / under)

    def fancy(self, portfolio: Portfolio) -> str:
        return format(self(portfolio), ".2f")

    @property
    def fancy_name(self) -> str:
        return "Omega Ratio"


@dataclass
class SortinoRatio:
    rf: float = 0.0
    annualizer: Optional[float] = None

    def __call__(self, portfolio: Portfolio) -> float:
        adjusted = adjust(portfolio.returns, self.rf)
        if self.annualizer is None:
            annualizer = extract_annualizer(adjusted)
        else:
            annualizer = self.annualizer

        returns_under_mar = np.clip(adjusted, a_min=-np.inf, a_max=0)
        downside_risk = np.sqrt((returns_under_mar ** 2).mean())

        return adjusted.mean() / downside_risk * np.sqrt(annualizer)

    def fancy(self, portfolio: Portfolio) -> str:
        return format(self(portfolio), ".2f")

    @property
    def fancy_name(self) -> str:
        return "Sortino Ratio"


@dataclass
class BenchmarkCorrelation:
    benchmark: Benchmark

    def __call__(self, portfolio: Portfolio) -> float:
        returns, benchmark = align(portfolio.returns, self.benchmark.returns)
        return returns.corr(benchmark)

    def fancy(self, portfolio: Portfolio) -> str:
        return format(self(portfolio), ".2f")

    @property
    def fancy_name(self) -> str:
        return f"{self.benchmark.name} Correlation"


@dataclass
class MeanExcessReturn:
    benchmark: Benchmark
    statistics: bool = False
    annualizer: Optional[float] = None

    def __call__(self, portfolio: Portfolio) -> float | Stats:
        adjusted = adjust(portfolio.returns, self.benchmark.returns)
        if self.annualizer is None:
            annualizer = extract_annualizer(adjusted)
        else:
            annualizer = self.annualizer

        mer = adjusted.mean() * annualizer

        if self.statistics:
            ttest = ttest_1samp(portfolio.returns, 0, alternative="greater")
            mer = stats_container_factory("MeanExcessReturn")(
                value=mer,
                t_stat=ttest.statistic,
                p_value=ttest.pvalue
            )

        return mer

    def fancy(self, portfolio: Portfolio) -> str:
        mer = self(portfolio)

        if self.statistics:
            return mer.template.format(
                value=mer.value * 100,
                stars="*" * mer.count_stars(),
                t_stat=mer.t_stat
            )

        return format(mer * 100, ".2f")

    @property
    def fancy_name(self) -> str:
        return "Mean Excess Return, %"


@dataclass
class Alpha:
    benchmark: Benchmark
    rf: float = 0.0
    statistics: bool = False
    annualizer: Optional[float] = None

    def __call__(self, portfolio: Portfolio) -> float | Stats:
        if self.annualizer is None:
            annualizer = extract_annualizer(portfolio.returns)
        else:
            annualizer = self.annualizer

        est = estimate_ols(portfolio.returns, self.benchmark.returns, self.rf)
        alpha = est.params[0] * annualizer

        if self.statistics:
            # TODO: t-stat and p-value for one-sided test
            alpha = stats_container_factory("Alpha")(
                value=alpha,
                p_value=est.pvalues[0],
                t_stat=est.tvalues[0]
            )

        return alpha

    def fancy(self, portfolio: Portfolio) -> str:
        alpha = self(portfolio)

        if self.statistics:
            return alpha.template.format(
                value=alpha.value * 100,
                stars="*" * alpha.count_stars(),
                t_stat=alpha.t_stat
            )

        return format(alpha * 100, ".2f")

    @property
    def fancy_name(self) -> str:
        return "Alpha, %"


@dataclass
class Beta:
    benchmark: Benchmark
    rf: float = 0.0
    statistics: bool = False

    def __call__(self, portfolio: Portfolio) -> float | Stats:
        est = estimate_ols(portfolio.returns, self.benchmark.returns, self.rf)
        beta = est.params[1]

        if self.statistics:
            beta = stats_container_factory("Beta")(
                value=beta,
                p_value=est.pvalues[1],
                t_stat=est.tvalues[1]
            )

        return beta

    def fancy(self, portfolio: Portfolio) -> str:
        beta = self(portfolio)

        if self.statistics:
            return beta.template.format(
                value=beta.value,
                stars="*" * beta.count_stars(),
                t_stat=beta.t_stat
            )

        return format(beta * 100, ".2f")

    @property
    def fancy_name(self) -> str:
        return "Beta"


@dataclass
class MeanTurnover:
    def __call__(self, portfolio: Portfolio) -> float:
        return Turnover()(portfolio).mean()

    def fancy(self, portfolio: Portfolio) -> str:
        return format(self(portfolio) * 100, ".2f")

    @property
    def fancy_name(self) -> str:
        return "Mean Turnover, %"
