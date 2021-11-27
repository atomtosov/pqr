from __future__ import annotations

from typing import Optional, Protocol

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

from pqr.core import Portfolio, Benchmark
from pqr.utils import align
from .utils import extract_annualizer, adjust, estimate_ols, stats_container_factory, estimate_rolling_ols

__all__ = [
    "CompoundedReturns", "Drawdown", "Turnover",
    "TotalReturn", "TrailingTotalReturn",
    "CAGR", "TrailingCAGR",
    "MeanReturn", "TrailingMeanReturn",
    "Volatility", "TrailingVolatility",
    "WinRate", "TrailingWinRate",
    "MaxDrawdown", "TrailingMaxDrawdown",
    "ValueAtRisk", "TrailingValueAtRisk",
    "ExpectedTailLoss", "TrailingExpectedTailLoss",
    "ExpectedTailReward", "TrailingExpectedTailReward",
    "RachevRatio", "TrailingRachevRatio",
    "CalmarRatio", "TrailingCalmarRatio",
    "SharpeRatio", "TrailingSharpeRatio",
    "OmegaRatio", "TrailingOmegaRatio",
    "SortinoRatio", "TrailingSortinoRatio",
    "BenchmarkCorrelation", "TrailingBenchmarkCorrelation",
    "MeanExcessReturn", "TrailingMeanExcessReturn",
    "Alpha", "TrailingAlpha",
    "Beta", "TrailingBeta",
    "MeanTurnover", "TrailingMeanTurnover",
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


class CompoundedReturns:
    def __call__(self, portfolio: Portfolio) -> pd.Series:
        return (1 + portfolio.returns).cumprod() - 1

    def fancy(self, portfolio: Portfolio) -> pd.Series:
        return self(portfolio) * 100

    @property
    def fancy_name(self) -> str:
        return "Compounded Returns, %"


class Drawdown:
    def __call__(self, portfolio: Portfolio) -> pd.Series:
        equity = CompoundedReturns()(portfolio) + 1
        high_water_mark = equity.cummax()
        return equity / high_water_mark - 1

    def fancy(self, portfolio: Portfolio) -> pd.Series:
        return self(portfolio) * 100

    @property
    def fancy_name(self) -> str:
        return "Drawdown, %"


class Turnover:
    def __call__(self, portfolio: Portfolio) -> pd.Series:
        positions = portfolio.positions.to_numpy()

        turnover = np.nansum(
            np.abs(np.diff(positions, axis=0)),
            axis=1
        )
        # add 1st period deals
        turnover = np.insert(turnover, 0, values=np.nansum(np.abs(positions[0])))

        return pd.Series(
            turnover,
            index=portfolio.positions.index
        )

    def fancy(self, portfolio: Portfolio) -> pd.Series:
        return self(portfolio) * 100

    @property
    def fancy_name(self) -> str:
        return "Turnover, %"


class TotalReturn:
    def __call__(self, portfolio: Portfolio) -> pd.Series:
        return CompoundedReturns()(portfolio).iat[-1]

    def fancy(self, portfolio: Portfolio) -> str:
        return format(self(portfolio) * 100, ".2f")

    @property
    def fancy_name(self) -> str:
        return "Total Return, %"


class TrailingTotalReturn:
    def __init__(self, window: Optional[int] = None):
        self.window = window

    def __call__(self, portfolio: Portfolio) -> pd.Series:
        if self.window is None:
            window = int(extract_annualizer(portfolio.returns))
        else:
            window = self.window

        return portfolio.returns.rolling(window).apply(
            lambda r: ((1 + r).cumprod() - 1).iloc[-1]
        ).iloc[window:]

    def fancy(self, portfolio: Portfolio) -> pd.Series:
        return self(portfolio) * 100

    @property
    def fancy_name(self) -> str:
        return "Trailing Total Return, %"


class CAGR:
    def __init__(self, annualizer: Optional[float] = None):
        self.annualizer = annualizer

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


class TrailingCAGR:
    def __init__(
            self,
            annualizer: Optional[float] = None,
            window: Optional[int] = None
    ):
        self.annualizer = annualizer
        self.window = window

    def __call__(self, portfolio: Portfolio) -> pd.Series:
        if self.annualizer is None:
            annualizer = extract_annualizer(portfolio.returns)
        else:
            annualizer = self.annualizer

        if self.window is None:
            window = int(extract_annualizer(portfolio.returns))
        else:
            window = self.window

        tr = TrailingTotalReturn(window)(portfolio)
        years = window / annualizer

        return (1 + tr) ** (1 / years) - 1

    def fancy(self, portfolio: Portfolio) -> pd.Series:
        return self(portfolio) * 100

    @property
    def fancy_name(self) -> str:
        return "Trailing CAGR, %"


class MeanReturn:
    def __init__(
            self,
            statistics: bool = False,
            annualizer: Optional[float] = None,
    ):
        self.statistics = statistics
        self.annualizer = annualizer

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


class TrailingMeanReturn:
    def __init__(
            self,
            statistics: bool = False,
            annualizer: Optional[float] = None,
            window: Optional[int] = None
    ):
        self.statistics = statistics
        self.annualizer = annualizer
        self.window = window

    def __call__(self, portfolio: Portfolio) -> pd.Series:
        if self.annualizer is None:
            annualizer = extract_annualizer(portfolio.returns)
        else:
            annualizer = self.annualizer

        if self.window is None:
            window = int(extract_annualizer(portfolio.returns))
        else:
            window = self.window

        return portfolio.returns.rolling(window).mean().iloc[window:] * annualizer

    def fancy(self, portfolio: Portfolio) -> pd.Series:
        return self(portfolio) * 100

    @property
    def fancy_name(self) -> str:
        return "Trailing Mean Return, %"


class Volatility:
    def __init__(self, annualizer: Optional[float] = None):
        self.annualizer = annualizer

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


class TrailingVolatility:
    def __init__(
            self,
            annualizer: Optional[float] = None,
            window: Optional[int] = None
    ):
        self.annualizer = annualizer
        self.window = window

    def __call__(self, portfolio: Portfolio) -> pd.Series:
        if self.annualizer is None:
            annualizer = extract_annualizer(portfolio.returns)
        else:
            annualizer = self.annualizer

        if self.window is None:
            window = int(extract_annualizer(portfolio.returns))
        else:
            window = self.window

        return portfolio.returns.rolling(window).std().iloc[window:] * np.sqrt(annualizer)

    def fancy(self, portfolio: Portfolio) -> pd.Series:
        return self(portfolio) * 100

    @property
    def fancy_name(self) -> str:
        return "Trailing Volatility, %"


class WinRate:
    def __call__(self, portfolio: Portfolio) -> float:
        return (portfolio.returns > 0).sum() / len(portfolio.returns)

    def fancy(self, portfolio: Portfolio) -> str:
        return format(self(portfolio) * 100, ".2f")

    @property
    def fancy_name(self) -> str:
        return "Win Rate, %"


class TrailingWinRate:
    def __init__(self, window: Optional[int] = None):
        self.window = window

    def __call__(self, portfolio: Portfolio) -> pd.Series:
        if self.window is None:
            window = int(extract_annualizer(portfolio.returns))
        else:
            window = self.window

        return (portfolio.returns > 0).rolling(window).sum().iloc[window:] / window

    def fancy(self, portfolio: Portfolio) -> pd.Series:
        return self(portfolio) * 100

    @property
    def fancy_name(self) -> str:
        return "Trailing Win Rate, %"


class MaxDrawdown:
    def __call__(self, portfolio: Portfolio) -> float:
        return Drawdown()(portfolio).min()

    def fancy(self, portfolio: Portfolio) -> str:
        return format(self(portfolio) * 100, ".2f")

    @property
    def fancy_name(self) -> str:
        return "Maximum Drawdown, %"


class TrailingMaxDrawdown:
    def __init__(self, window: Optional[int] = None):
        self.window = window

    def __call__(self, portfolio: Portfolio) -> pd.Series:
        if self.window is None:
            window = int(extract_annualizer(portfolio.returns))
        else:
            window = self.window

        return portfolio.returns.rolling(window).apply(
            lambda r: ((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()
        ).iloc[window:]

    def fancy(self, portfolio: Portfolio) -> pd.Series:
        return self(portfolio) * 100

    @property
    def fancy_name(self) -> str:
        return "Trailing Maximum Drawdown, %"


class ValueAtRisk:
    def __init__(
            self,
            cutoff: float = 0.05,
            annualizer: Optional[float] = None
    ):
        self.cutoff = cutoff
        self.annualizer = annualizer

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


class TrailingValueAtRisk:
    def __init__(
            self,
            cutoff: float = 0.05,
            annualizer: Optional[float] = None,
            window: Optional[int] = None
    ):
        self.cutoff = cutoff
        self.annualizer = annualizer
        self.window = window

    def __call__(self, portfolio: Portfolio) -> pd.Series:
        if self.annualizer is None:
            annualizer = extract_annualizer(portfolio.returns)
        else:
            annualizer = self.annualizer

        if self.window is None:
            window = int(extract_annualizer(portfolio.returns))
        else:
            window = self.window

        return portfolio.returns.rolling(window).quantile(self.cutoff) * np.sqrt(annualizer)

    def fancy(self, portfolio: Portfolio) -> pd.Series:
        return self(portfolio) * 100

    @property
    def fancy_name(self) -> str:
        return "Trailing Value at Risk, %"


class ExpectedTailLoss:
    def __init__(
            self,
            cutoff: float = 0.05,
            annualizer: Optional[float] = None
    ):
        self.cutoff = cutoff
        self.annualizer = annualizer

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


class TrailingExpectedTailLoss:
    def __init__(
            self,
            cutoff: float = 0.05,
            annualizer: Optional[float] = None,
            window: Optional[int] = None
    ):
        self.cutoff = cutoff
        self.annualizer = annualizer
        self.window = window

    def __call__(self, portfolio: Portfolio) -> pd.Series:
        if self.annualizer is None:
            annualizer = extract_annualizer(portfolio.returns)
        else:
            annualizer = self.annualizer

        if self.window is None:
            window = int(extract_annualizer(portfolio.returns))
        else:
            window = self.window

        return portfolio.returns.rolling(window).apply(
            lambda r: r[r <= r.quantile(self.cutoff)].mean()
        ).iloc[window:] * np.sqrt(annualizer)

    def fancy(self, portfolio: Portfolio) -> pd.Series:
        return self(portfolio) * 100

    @property
    def fancy_name(self) -> str:
        return "Trailing Expected Tail Loss, %"


class ExpectedTailReward:
    def __init__(
            self,
            cutoff: float = 0.95,
            annualizer: Optional[float] = None
    ):
        self.cutoff = cutoff
        self.annualizer = annualizer

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


class TrailingExpectedTailReward:
    def __init__(
            self,
            cutoff: float = 0.95,
            annualizer: Optional[float] = None,
            window: Optional[int] = None
    ):
        self.cutoff = cutoff
        self.annualizer = annualizer
        self.window = window

    def __call__(self, portfolio: Portfolio) -> pd.Series:
        if self.annualizer is None:
            annualizer = extract_annualizer(portfolio.returns)
        else:
            annualizer = self.annualizer

        if self.window is None:
            window = int(extract_annualizer(portfolio.returns))
        else:
            window = self.window

        return portfolio.returns.rolling(window).apply(
            lambda r: r[r >= r.quantile(self.cutoff)].mean()
        ).iloc[window:] * np.sqrt(annualizer)

    def fancy(self, portfolio: Portfolio) -> pd.Series:
        return self(portfolio) * 100

    @property
    def fancy_name(self) -> str:
        return "Trailing Expected Tail Reward, %"


class RachevRatio:
    def __init__(
            self,
            reward_cutoff: float = 0.95,
            risk_cutoff: float = 0.05
    ):
        self.reward_cutoff = reward_cutoff
        self.risk_cutoff = risk_cutoff

    def __call__(self, portfolio: Portfolio) -> float:
        etr = ExpectedTailReward(self.reward_cutoff)(portfolio)
        etl = ExpectedTailLoss(self.risk_cutoff)(portfolio)
        return -(etr / etl)

    def fancy(self, portfolio: Portfolio) -> str:
        return format(self(portfolio), ".2f")

    @property
    def fancy_name(self) -> str:
        return "Rachev Ratio"


class TrailingRachevRatio:
    def __init__(
            self,
            reward_cutoff: float = 0.95,
            risk_cutoff: float = 0.05,
            window: Optional[int] = None
    ):
        self.reward_cutoff = reward_cutoff
        self.risk_cutoff = risk_cutoff
        self.window = window

    def __call__(self, portfolio: Portfolio) -> pd.Series:
        if self.window is None:
            window = int(extract_annualizer(portfolio.returns))
        else:
            window = self.window

        etr = TrailingExpectedTailReward(self.reward_cutoff, window)(portfolio)
        etl = TrailingExpectedTailLoss(self.risk_cutoff, window)(portfolio)
        return -(etr / etl)

    def fancy(self, portfolio: Portfolio) -> pd.Series:
        return self(portfolio)

    @property
    def fancy_name(self) -> str:
        return "Trailing Rachev Ratio"


class CalmarRatio:
    def __init__(self, annualizer: Optional[float] = None):
        self.annualizer = annualizer

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


class TrailingCalmarRatio:
    def __init__(
            self,
            annualizer: Optional[float] = None,
            window: Optional[int] = None
    ):
        self.annualizer = annualizer
        self.window = window

    def __call__(self, portfolio: Portfolio) -> pd.Series:
        if self.annualizer is None:
            annualizer = extract_annualizer(portfolio.returns)
        else:
            annualizer = self.annualizer

        if self.window is None:
            window = int(extract_annualizer(portfolio.returns))
        else:
            window = self.window

        return -(TrailingCAGR(annualizer, window)(portfolio) / TrailingMaxDrawdown(window)(portfolio))

    def fancy(self, portfolio: Portfolio) -> pd.Series:
        return self(portfolio)

    @property
    def fancy_name(self) -> str:
        return "Trailing Calmar Ratio"


class SharpeRatio:
    def __init__(
            self,
            rf: float = 0.0,
            annualizer: Optional[float] = None
    ):
        self.rf = rf
        self.annualizer = annualizer

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


class TrailingSharpeRatio:
    def __init__(
            self,
            rf: float = 0.0,
            annualizer: Optional[float] = None,
            window: Optional[int] = None
    ):
        self.rf = rf
        self.annualizer = annualizer
        self.window = window

    def __call__(self, portfolio: Portfolio) -> pd.Series:
        adjusted = adjust(portfolio.returns, self.rf)
        if self.annualizer is None:
            annualizer = extract_annualizer(adjusted)
        else:
            annualizer = self.annualizer

        if self.window is None:
            window = int(extract_annualizer(portfolio.returns))
        else:
            window = self.window

        mr = adjusted.rolling(window).mean().iloc[window:]
        std = adjusted.rolling(window).std().iloc[window:]

        return mr / std * np.sqrt(annualizer)

    def fancy(self, portfolio: Portfolio) -> pd.Series:
        return self(portfolio)

    @property
    def fancy_name(self) -> str:
        return "Trailing Sharpe Ratio"


class OmegaRatio:
    def __init__(self, rf: float = 0.0):
        self.rf = rf

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


class TrailingOmegaRatio:
    def __init__(
            self,
            rf: float = 0.0,
            window: Optional[int] = None
    ):
        self.rf = rf
        self.window = window

    def __call__(self, portfolio: Portfolio) -> pd.Series:
        if self.window is None:
            window = int(extract_annualizer(portfolio.returns))
        else:
            window = self.window

        adjusted = adjust(portfolio.returns, self.rf)

        return adjusted.rolling(window).apply(
            lambda r: -(r[r > 0].sum() / r[r < 0].sum())
        )

    def fancy(self, portfolio: Portfolio) -> pd.Series:
        return self(portfolio)

    @property
    def fancy_name(self) -> str:
        return "Omega Ratio"


class SortinoRatio:
    def __init__(
            self,
            rf: float = 0.0,
            annualizer: Optional[float] = None
    ):
        self.rf = rf
        self.annualizer = annualizer

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


class TrailingSortinoRatio:
    def __init__(
            self,
            rf: float = 0.0,
            annualizer: Optional[float] = None,
            window: Optional[int] = None
    ):
        self.rf = rf
        self.annualizer = annualizer
        self.window = window

    def __call__(self, portfolio: Portfolio) -> pd.Series:
        adjusted = adjust(portfolio.returns, self.rf)
        if self.annualizer is None:
            annualizer = extract_annualizer(adjusted)
        else:
            annualizer = self.annualizer

        if self.window is None:
            window = int(extract_annualizer(portfolio.returns))
        else:
            window = self.window

        mr = adjusted.rolling(window).mean().iloc[window:]
        downside_risk = adjusted.rolling(window).apply(
            lambda r: np.sqrt((np.clip(adjusted, a_min=-np.inf, a_max=0) ** 2).mean())
        ).iloc[window:]

        return mr / downside_risk * np.sqrt(annualizer)

    def fancy(self, portfolio: Portfolio) -> pd.Series:
        return self(portfolio)

    @property
    def fancy_name(self) -> str:
        return "Trailing Sortino Ratio"


class BenchmarkCorrelation:
    def __init__(self, benchmark: Benchmark):
        self.benchmark = benchmark

    def __call__(self, portfolio: Portfolio) -> float:
        returns, benchmark = align(portfolio.returns, self.benchmark.returns)
        return returns.corr(benchmark)

    def fancy(self, portfolio: Portfolio) -> str:
        return format(self(portfolio), ".2f")

    @property
    def fancy_name(self) -> str:
        return f"{self.benchmark.name} Correlation"


class TrailingBenchmarkCorrelation:
    def __init__(
            self,
            benchmark: Benchmark,
            window: Optional[int] = None
    ):
        self.benchmark = benchmark
        self.window = window

    def __call__(self, portfolio: Portfolio) -> pd.Series:
        returns, benchmark = align(portfolio.returns, self.benchmark.returns)

        if self.window is None:
            window = int(extract_annualizer(returns))
        else:
            window = self.window

        return returns.rolling(window).corr(benchmark)

    def fancy(self, portfolio: Portfolio) -> pd.Series:
        return self(portfolio)

    @property
    def fancy_name(self) -> str:
        return f"Trailing {self.benchmark.name} Correlation"


class MeanExcessReturn:
    def __init__(
            self,
            benchmark: Benchmark,
            statistics: bool = False,
            annualizer: Optional[float] = None
    ):
        self.benchmark = benchmark
        self.statistics = statistics
        self.annualizer = annualizer

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


class TrailingMeanExcessReturn:
    def __init__(
            self,
            benchmark: Benchmark,
            annualizer: Optional[float] = None,
            window: Optional[int] = None
    ):
        self.benchmark = benchmark
        self.annualizer = annualizer
        self.window = window

    def __call__(self, portfolio: Portfolio) -> pd.Series:
        adjusted = adjust(portfolio.returns, self.benchmark.returns)

        if self.annualizer is None:
            annualizer = extract_annualizer(adjusted)
        else:
            annualizer = self.annualizer

        if self.window is None:
            window = int(extract_annualizer(adjusted))
        else:
            window = self.window

        return adjusted.rolling(window).mean().iloc[window:] * annualizer

    def fancy(self, portfolio: Portfolio) -> pd.Series:
        return self(portfolio) * 100

    @property
    def fancy_name(self) -> str:
        return "Trailing Mean Excess Return, %"


class Alpha:
    def __init__(
            self,
            benchmark: Benchmark,
            rf: float = 0.0,
            statistics: bool = False,
            annualizer: Optional[float] = None
    ):
        self.benchmark = benchmark
        self.rf = rf
        self.statistics = statistics
        self.annualizer = annualizer

    def __call__(self, portfolio: Portfolio) -> float | Stats:
        if self.annualizer is None:
            annualizer = extract_annualizer(portfolio.returns)
        else:
            annualizer = self.annualizer

        returns, benchmark = align(portfolio.returns, self.benchmark.returns)
        est = estimate_ols(returns, benchmark, self.rf)
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


class TrailingAlpha:
    def __init__(
            self,
            benchmark: Benchmark,
            rf: float = 0.0,
            annualizer: Optional[float] = None,
            window: Optional[int] = None
    ):
        self.benchmark = benchmark
        self.rf = rf
        self.annualizer = annualizer
        self.window = window

    def __call__(self, portfolio: Portfolio) -> pd.Series:
        if self.annualizer is None:
            annualizer = extract_annualizer(portfolio.returns)
        else:
            annualizer = self.annualizer

        if self.window is None:
            window = int(extract_annualizer(portfolio.returns))
        else:
            window = self.window

        returns, benchmark = align(portfolio.returns, self.benchmark.returns)

        return pd.Series(
            estimate_rolling_ols(
                returns,
                benchmark,
                window,
                self.rf
            ).params[window:, 0] * annualizer,
            index=returns.index[window:].copy()
        )

    def fancy(self, portfolio: Portfolio) -> pd.Series:
        return self(portfolio) * 100

    @property
    def fancy_name(self) -> str:
        return "Trailing Alpha, %"


class Beta:
    def __init__(
            self,
            benchmark: Benchmark,
            rf: float = 0.0,
            statistics: bool = False,
    ):
        self.benchmark = benchmark
        self.rf = rf
        self.statistics = statistics

    def __call__(self, portfolio: Portfolio) -> float | Stats:
        returns, benchmark = align(portfolio.returns, self.benchmark.returns)
        est = estimate_ols(returns, benchmark, self.rf)
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

        return format(beta, ".2f")

    @property
    def fancy_name(self) -> str:
        return "Beta"


class TrailingBeta:
    def __init__(
            self,
            benchmark: Benchmark,
            rf: float = 0.0,
            window: Optional[int] = None
    ):
        self.benchmark = benchmark
        self.rf = rf
        self.window = window

    def __call__(self, portfolio: Portfolio) -> pd.Series:
        if self.window is None:
            window = int(extract_annualizer(portfolio.returns))
        else:
            window = self.window

        returns, benchmark = align(portfolio.returns, self.benchmark.returns)

        return pd.Series(
            estimate_rolling_ols(
                returns, benchmark, window, self.rf
            ).params[window:, 1],
            index=returns.index[window:].copy()
        )

    def fancy(self, portfolio: Portfolio) -> pd.Series:
        return self(portfolio)

    @property
    def fancy_name(self) -> str:
        return "Trailing Beta, %"


class MeanTurnover:
    def __init__(self, annualizer: Optional[float] = None):
        self.annualizer = annualizer

    def __call__(self, portfolio: Portfolio) -> float:
        if self.annualizer is None:
            annualizer = extract_annualizer(portfolio.returns)
        else:
            annualizer = self.annualizer

        return Turnover()(portfolio).mean() * annualizer

    def fancy(self, portfolio: Portfolio) -> str:
        return format(self(portfolio) * 100, ".2f")

    @property
    def fancy_name(self) -> str:
        return "Mean Turnover, %"


class TrailingMeanTurnover:
    def __init__(
            self,
            annualizer: Optional[float] = None,
            window: Optional[int] = None
    ):
        self.annualizer = annualizer
        self.window = window

    def __call__(self, portfolio: Portfolio) -> pd.Series:
        if self.annualizer is None:
            annualizer = extract_annualizer(portfolio.returns)
        else:
            annualizer = self.annualizer

        if self.window is None:
            window = int(extract_annualizer(portfolio.returns))
        else:
            window = self.window

        return Turnover()(portfolio).rolling(window).mean().iloc[window:] * annualizer

    def fancy(self, portfolio: Portfolio) -> pd.Series:
        return self(portfolio) * 100

    @property
    def fancy_name(self) -> str:
        return "Trailing Mean Turnover, %"
