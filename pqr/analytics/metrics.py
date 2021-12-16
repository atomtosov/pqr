from __future__ import annotations

from dataclasses import dataclass, field, make_dataclass
from typing import Optional, Protocol, runtime_checkable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import ttest_1samp
from statsmodels.regression.rolling import RollingOLS

from pqr.core import Portfolio, Benchmark
from pqr.utils import align, extract_annualizer, adjust

__all__ = [
    "NumericMetric", "TimeSeriesMetric",

    "CompoundedReturns", "TotalReturn", "CAGR",
    "Drawdown", "MaxDrawdown",
    "Turnover", "MeanTurnover",

    "MeanReturn",
    "Volatility",
    "WinRate",
    "CalmarRatio",
    "SharpeRatio",
    "BenchmarkCorrelation",
    "MeanExcessReturn",
    "Alpha", "Beta",
]


@runtime_checkable
class NumericMetric(Protocol):
    window: Optional[int]
    name: str

    def calculate(self, portfolio: Portfolio) -> float | Stats:
        pass

    def trailing(self, portfolio: Portfolio) -> pd.Series:
        pass

    def fancy(self, portfolio: Portfolio) -> str:
        pass


@runtime_checkable
class TimeSeriesMetric(Protocol):
    name: str

    def calculate(self, portfolio: Portfolio) -> pd.Series:
        pass


@dataclass
class CompoundedReturns:
    name: str = field(init=False, default="Compounded Returns")

    def calculate(self, portfolio: Portfolio) -> pd.Series:
        return (1 + portfolio.returns).cumprod() - 1


@dataclass
class Drawdown:
    name: str = field(init=False, default="Drawdown")

    def calculate(self, portfolio: Portfolio) -> pd.Series:
        equity = CompoundedReturns().calculate(portfolio) + 1
        high_water_mark = equity.cummax()
        return equity / high_water_mark - 1


@dataclass
class Turnover:
    name: str = field(init=False, default="Turnover")

    def calculate(self, portfolio: Portfolio) -> pd.Series:
        positions = portfolio.positions.to_numpy()

        positions_change = np.diff(positions, axis=0)
        turnover = np.nansum(np.abs(positions_change), axis=1)

        # add 1st period deals
        turnover = np.insert(turnover, 0,
                             values=np.nansum(np.abs(positions[0])))

        return pd.Series(turnover,
                         index=portfolio.positions.index.copy())


@dataclass
class TotalReturn:
    window: Optional[int] = None

    name: str = field(init=False, default="Total Return, %")

    def calculate(self, portfolio: Portfolio) -> float:
        return CompoundedReturns().calculate(portfolio).iat[-1]

    def trailing(self, portfolio: Portfolio) -> pd.Series:
        if self.window is None:
            window = int(extract_annualizer(portfolio.returns))
        else:
            window = self.window

        return portfolio.returns.rolling(window).apply(
            lambda r: ((1 + r).cumprod() - 1).iat[-1]
        ).iloc[window:]

    def fancy(self, portfolio: Portfolio) -> str:
        return format(self.calculate(portfolio) * 100, ".2f")


@dataclass
class CAGR:
    annualizer: Optional[float] = None
    window: Optional[int] = None

    name: str = field(init=False, default="CAGR, %")

    def calculate(self, portfolio: Portfolio) -> float:
        if self.annualizer is None:
            annualizer = extract_annualizer(portfolio.returns)
        else:
            annualizer = self.annualizer

        tr = TotalReturn().calculate(portfolio)
        years = len(portfolio.returns) / annualizer

        return (1 + tr) ** (1 / years) - 1

    def trailing(self, portfolio: Portfolio) -> pd.Series:
        if self.annualizer is None:
            annualizer = extract_annualizer(portfolio.returns)
        else:
            annualizer = self.annualizer

        if self.window is None:
            window = int(extract_annualizer(portfolio.returns))
        else:
            window = self.window

        tr = TotalReturn(window).trailing(portfolio)
        years = window / annualizer

        return (1 + tr) ** (1 / years) - 1

    def fancy(self, portfolio: Portfolio) -> str:
        return format(self.calculate(portfolio) * 100, ".2f")


@dataclass
class MeanReturn:
    statistics: bool = False
    annualizer: Optional[float] = None
    window: Optional[int] = None

    name: str = field(init=False, default="Mean Return, %")

    def calculate(self, portfolio: Portfolio) -> float | Stats:
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
                p_value=ttest.pvalue)

        return mr

    def trailing(self, portfolio: Portfolio) -> pd.Series:
        if self.annualizer is None:
            annualizer = extract_annualizer(portfolio.returns)
        else:
            annualizer = self.annualizer

        if self.window is None:
            window = int(extract_annualizer(portfolio.returns))
        else:
            window = self.window

        return portfolio.returns.rolling(window).mean().iloc[window:] * annualizer

    def fancy(self, portfolio: Portfolio) -> str:
        mr = self.calculate(portfolio)
        if self.statistics:
            return mr.template.format(
                value=mr.value * 100,
                stars="*" * mr.count_stars(),
                t_stat=mr.t_stat)

        return format(mr * 100, ".2f")


@dataclass
class Volatility:
    annualizer: Optional[float] = None
    window: Optional[int] = None

    name: str = field(init=False, default="Volatility, %")

    def calculate(self, portfolio: Portfolio) -> float:
        if self.annualizer is None:
            annualizer = extract_annualizer(portfolio.returns)
        else:
            annualizer = self.annualizer

        return portfolio.returns.std() * np.sqrt(annualizer)

    def trailing(self, portfolio: Portfolio) -> pd.Series:
        if self.annualizer is None:
            annualizer = extract_annualizer(portfolio.returns)
        else:
            annualizer = self.annualizer

        if self.window is None:
            window = int(extract_annualizer(portfolio.returns))
        else:
            window = self.window

        return portfolio.returns.rolling(window).std().iloc[window:] * np.sqrt(annualizer)

    def fancy(self, portfolio: Portfolio) -> str:
        return format(self.calculate(portfolio) * 100, ".2f")


@dataclass
class WinRate:
    window: Optional[int] = None

    name: str = field(init=False, default="Win Rate, %")

    def calculate(self, portfolio: Portfolio) -> float:
        return (portfolio.returns > 0).sum() / len(portfolio.returns)

    def trailing(self, portfolio: Portfolio) -> pd.Series:
        if self.window is None:
            window = int(extract_annualizer(portfolio.returns))
        else:
            window = self.window

        return (portfolio.returns > 0).rolling(window).sum().iloc[window:] / window

    def fancy(self, portfolio: Portfolio) -> str:
        return format(self.calculate(portfolio), ".2f")


@dataclass
class MaxDrawdown:
    window: Optional[int] = None

    name: str = field(init=False, default="Maximum Drawdown, %")

    def calculate(self, portfolio: Portfolio) -> float:
        return Drawdown().calculate(portfolio).min()

    def trailing(self, portfolio: Portfolio) -> float:
        if self.window is None:
            window = int(extract_annualizer(portfolio.returns))
        else:
            window = self.window

        return portfolio.returns.rolling(window).apply(
            lambda r: ((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()
        ).iloc[window:]

    def fancy(self, portfolio: Portfolio) -> str:
        return format(self.calculate(portfolio), ".2f")


@dataclass
class CalmarRatio:
    annualizer: Optional[float] = None
    window: Optional[int] = None

    name: str = field(init=False, default="Calmar Ratio")

    def calculate(self, portfolio: Portfolio) -> float:
        if self.annualizer is None:
            annualizer = extract_annualizer(portfolio.returns)
        else:
            annualizer = self.annualizer

        cagr = CAGR(annualizer).calculate(portfolio)
        max_dd = MaxDrawdown().calculate(portfolio)

        return -cagr / max_dd

    def trailing(self, portfolio: Portfolio) -> pd.Series:
        if self.annualizer is None:
            annualizer = extract_annualizer(portfolio.returns)
        else:
            annualizer = self.annualizer

        if self.window is None:
            window = int(extract_annualizer(portfolio.returns))
        else:
            window = self.window

        cagr = CAGR(annualizer, window).trailing(portfolio)
        max_dd = MaxDrawdown(window).trailing(portfolio)

        return -cagr / max_dd

    def fancy(self, portfolio: Portfolio) -> str:
        return format(self.calculate(portfolio), ".2f")


@dataclass
class SharpeRatio:
    rf: float = 0.0
    annualizer: Optional[float] = None
    window: Optional[int] = None

    name: str = field(init=False, default="Sharpe Ratio")

    def calculate(self, portfolio: Portfolio) -> float:
        adjusted = adjust(portfolio.returns, self.rf)
        if self.annualizer is None:
            annualizer = extract_annualizer(adjusted)
        else:
            annualizer = self.annualizer

        return adjusted.mean() / adjusted.std() * np.sqrt(annualizer)

    def trailing(self, portfolio: Portfolio) -> pd.Series:
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

    def fancy(self, portfolio: Portfolio) -> str:
        return format(self.calculate(portfolio), ".2f")


@dataclass
class BenchmarkCorrelation:
    benchmark: Benchmark
    window: Optional[int] = None

    name: str = field(init=False, default="Benchmark Correlation")

    def calculate(self, portfolio: Portfolio) -> float:
        returns, benchmark = align(portfolio.returns, self.benchmark.returns)
        return returns.corr(benchmark)

    def trailing(self, portfolio: Portfolio) -> pd.Series:
        returns, benchmark = align(portfolio.returns, self.benchmark.returns)

        if self.window is None:
            window = int(extract_annualizer(returns))
        else:
            window = self.window

        return returns.rolling(window).corr(benchmark)

    def fancy(self, portfolio: Portfolio) -> str:
        return format(self.calculate(portfolio), ".2f")


@dataclass
class MeanExcessReturn:
    benchmark: Benchmark
    statistics: bool = False
    annualizer: Optional[float] = None
    window: Optional[int] = None

    name: str = field(init=False, default="Mean Excess Return, %")

    def calculate(self, portfolio: Portfolio) -> float | Stats:
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
                p_value=ttest.pvalue)

        return mer

    def trailing(self, portfolio: Portfolio) -> pd.Series:
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

    def fancy(self, portfolio: Portfolio) -> str:
        mer = self.calculate(portfolio)

        if self.statistics:
            return mer.template.format(
                value=mer.value * 100,
                stars="*" * mer.count_stars(),
                t_stat=mer.t_stat)

        return format(mer * 100, ".2f")


@dataclass
class Alpha:
    benchmark: Benchmark
    rf: float = 0.0
    statistics: bool = False
    annualizer: Optional[float] = None
    window: Optional[int] = None

    name: str = field(init=False, default="Alpha, %")

    def calculate(self, portfolio: Portfolio) -> float | Stats:
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
                t_stat=est.tvalues[0])

        return alpha

    def trailing(self, portfolio: Portfolio) -> pd.Series:
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

    def fancy(self, portfolio: Portfolio) -> str:
        alpha = self.calculate(portfolio)

        if self.statistics:
            return alpha.template.format(
                value=alpha.value * 100,
                stars="*" * alpha.count_stars(),
                t_stat=alpha.t_stat)

        return format(alpha * 100, ".2f")


@dataclass
class Beta:
    benchmark: Benchmark
    rf: float = 0.0
    statistics: bool = False
    window: Optional[int] = None

    name: str = field(init=False, default="Beta")

    def calculate(self, portfolio: Portfolio) -> float | Stats:
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

    def trailing(self, portfolio: Portfolio) -> pd.Series:
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

    def fancy(self, portfolio: Portfolio) -> str:
        beta = self.calculate(portfolio)

        if self.statistics:
            return beta.template.format(
                value=beta.value,
                stars="*" * beta.count_stars(),
                t_stat=beta.t_stat)

        return format(beta, ".2f")


@dataclass
class MeanTurnover:
    annualizer: Optional[float] = None
    window: Optional[int] = None

    def calculate(self, portfolio: Portfolio) -> float:
        if self.annualizer is None:
            annualizer = extract_annualizer(portfolio.returns)
        else:
            annualizer = self.annualizer

        return Turnover().calculate(portfolio).mean() * annualizer

    def trailing(self, portfolio: Portfolio) -> pd.Series:
        if self.annualizer is None:
            annualizer = extract_annualizer(portfolio.returns)
        else:
            annualizer = self.annualizer

        if self.window is None:
            window = int(extract_annualizer(portfolio.returns))
        else:
            window = self.window

        return Turnover().calculate(portfolio).rolling(window).mean().iloc[window:] * annualizer

    def fancy(self, portfolio: Portfolio) -> str:
        return format(self.calculate(portfolio) * 100, ".2f")


class Stats(Protocol):
    value: float
    t_stat: float
    p_value: float

    template: str

    def count_stars(self) -> int:
        pass


def stats_container_factory(metric_name: str) -> type:
    return make_dataclass(
        metric_name,
        [
            ("value", float),
            ("t_stat", float),
            ("p_value", float),
        ],
        namespace={
            "template": property(lambda self: "{value:.2f}{stars} ({t_stat:.2f})"),
            "count_stars": lambda self: 3 if self.p_value < 0.01 else (
                2 if self.p_value < 0.05 else (
                    1 if self.p_value < 0.1 else 0))
        }
    )


def estimate_ols(
        returns: pd.Series,
        benchmark: pd.Series,
        rf: float = 0.0
):
    adjusted_returns = adjust(returns, rf)
    adjusted_benchmark = adjust(benchmark, rf)

    y, x = align(adjusted_returns, adjusted_benchmark)
    x = sm.add_constant(x.to_numpy())
    ols = sm.OLS(y.to_numpy(), x)

    return ols.fit()


def estimate_rolling_ols(
        returns: pd.Series,
        benchmark: pd.Series,
        window: int,
        rf: float = 0.0,
):
    adjusted_returns = adjust(returns, rf)
    adjusted_benchmark = adjust(benchmark, rf)

    y, x = align(adjusted_returns, adjusted_benchmark)
    x = sm.add_constant(x.to_numpy())
    ols = RollingOLS(y.to_numpy(), x, window=window)

    return ols.fit()
