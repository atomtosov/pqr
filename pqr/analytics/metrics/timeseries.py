from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from pqr.analytics.utils import extract_annualizer, adjust, estimate_ols
from pqr.core import Portfolio, Benchmark
from pqr.utils import align_many, align


__all__ = [
    "CompoundedReturns",
    "Drawdown",
    "Turnover",
    "TrailingTotalReturn",
    "TrailingCAGR",
    "TrailingMeanReturn",
    "TrailingVolatility",
    "TrailingWinRate",
    "TrailingMaxDrawdown",
    "TrailingValueAtRisk",
    "TrailingExpectedTailLoss",
    "TrailingExpectedTailReward",
    "TrailingRachevRatio",
    "TrailingCalmarRatio",
    "TrailingSharpeRatio",
    "TrailingSortinoRatio",
    "TrailingMeanExcessReturn",
    "TrailingAlpha",
    "TrailingBeta",
    "TrailingMeanTurnover",
]


@dataclass
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


@dataclass
class CompoundedReturns:
    def __call__(self, portfolio: Portfolio) -> pd.Series:
        return (1 + portfolio.returns).cumprod() - 1

    def fancy(self, portfolio: Portfolio) -> pd.Series:
        return self(portfolio) * 100

    @property
    def fancy_name(self) -> str:
        return "Compounded Returns, %"


@dataclass
class Turnover:
    def __call__(self, portfolio: Portfolio) -> pd.Series:
        # TODO: change it
        longs, shorts = portfolio.get_longs(), portfolio.get_shorts()
        longs, shorts, positions = align_many(longs, shorts, portfolio.positions)

        turnover_long = np.nansum(longs.diff().abs(), axis=0)
        turnover_short = np.nansum(shorts.diff().abs(), axis=0)

        return pd.Series(
            turnover_long + turnover_short,
            index=positions.index
        )

    def fancy(self, portfolio: Portfolio) -> pd.Series:
        return self(portfolio) * 100

    @property
    def fancy_name(self) -> str:
        return "Turnover, %"


@dataclass
class TrailingTotalReturn:
    window: Optional[int] = None

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


@dataclass
class TrailingCAGR:
    window: Optional[int] = None
    annualizer: Optional[float] = None

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


@dataclass
class TrailingMeanReturn:
    window: Optional[int] = None
    annualizer: Optional[float] = None

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


@dataclass
class TrailingVolatility:
    window: Optional[int] = None
    annualizer: Optional[float] = None

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
        return "Volatility, %"


@dataclass
class TrailingWinRate:
    window: Optional[int] = None

    def __call__(self, portfolio: Portfolio) -> pd.Series:
        if self.window is None:
            window = int(extract_annualizer(portfolio.returns))
        else:
            window = self.window

        return portfolio.returns.rolling(window).apply(
            lambda r: (r > 0).sum() / window
        ).iloc[window:]

    def fancy(self, portfolio: Portfolio) -> pd.Series:
        return self(portfolio) * 100

    @property
    def fancy_name(self) -> str:
        return "Trailing Win Rate, %"


@dataclass
class TrailingMaxDrawdown:
    window: Optional[int] = None

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


@dataclass
class TrailingValueAtRisk:
    cutoff: float = 0.05
    window: Optional[int] = None
    annualizer: Optional[float] = None

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


@dataclass
class TrailingExpectedTailLoss:
    cutoff: float = 0.05
    window: Optional[int] = None
    annualizer: Optional[float] = None

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


@dataclass
class TrailingExpectedTailReward:
    cutoff: float = 0.95
    window: Optional[int] = None
    annualizer: Optional[float] = None

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


@dataclass
class TrailingRachevRatio:
    reward_cutoff: float = 0.95
    risk_cutoff: float = 0.05
    window: Optional[int] = None

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


@dataclass
class TrailingCalmarRatio:
    window: Optional[int] = None
    annualizer: Optional[float] = None

    def __call__(self, portfolio: Portfolio) -> pd.Series:
        if self.annualizer is None:
            annualizer = extract_annualizer(portfolio.returns)
        else:
            annualizer = self.annualizer

        if self.window is None:
            window = int(extract_annualizer(portfolio.returns))
        else:
            window = self.window

        return -(TrailingCAGR(window, annualizer)(portfolio) / TrailingMaxDrawdown(window)(portfolio))

    def fancy(self, portfolio: Portfolio) -> pd.Series:
        return self(portfolio)

    @property
    def fancy_name(self) -> str:
        return "Trailing Calmar Ratio"


@dataclass
class TrailingSharpeRatio:
    rf: float = 0.0
    window: Optional[int] = None
    annualizer: Optional[float] = None

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


@dataclass
class TrailingSortinoRatio:
    rf: float = 0.0
    window: Optional[int] = None
    annualizer: Optional[float] = None

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


@dataclass
class BenchmarkCorrelation:
    benchmark: Benchmark
    window: Optional[int] = None

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


@dataclass
class TrailingMeanExcessReturn:
    benchmark: Benchmark
    window: Optional[int] = None

    def __call__(self, portfolio: Portfolio) -> pd.Series:
        adjusted = adjust(portfolio.returns, self.benchmark.returns)

        if self.window is None:
            window = int(extract_annualizer(adjusted))
        else:
            window = self.window

        return adjusted.rolling(window).mean().iloc[window:]

    def fancy(self, portfolio: Portfolio) -> pd.Series:
        return self(portfolio) * 100

    @property
    def fancy_name(self) -> str:
        return "Trailing Mean Excess Return, %"


@dataclass
class TrailingAlpha:
    benchmark: Benchmark
    rf: float = 0.0
    window: Optional[int] = None
    annualizer: Optional[float] = None

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
            estimate_ols(portfolio.returns, self.benchmark.returns, self.rf).params[0]
        ).iloc[window:] * annualizer

    def fancy(self, portfolio: Portfolio) -> pd.Series:
        return self(portfolio) * 100

    @property
    def fancy_name(self) -> str:
        return "Trailing Alpha, %"


@dataclass
class TrailingBeta:
    benchmark: Benchmark
    rf: float = 0.0
    window: Optional[int] = None

    def __call__(self, portfolio: Portfolio) -> pd.Series:
        if self.window is None:
            window = int(extract_annualizer(portfolio.returns))
        else:
            window = self.window

        return portfolio.returns.rolling(window).apply(
            estimate_ols(portfolio.returns, self.benchmark.returns, self.rf).params[1]
        ).iloc[window:]

    def fancy(self, portfolio: Portfolio) -> pd.Series:
        return self(portfolio)

    @property
    def fancy_name(self) -> str:
        return "Trailing Beta, %"


@dataclass
class TrailingMeanTurnover:
    window: Optional[int] = None

    def __call__(self, portfolio: Portfolio) -> pd.Series:
        if self.window is None:
            window = int(extract_annualizer(portfolio.returns))
        else:
            window = self.window

        return Turnover()(portfolio).rolling(window).mean().iloc[window:]

    def fancy(self, portfolio: Portfolio) -> pd.Series:
        return self(portfolio) * 100

    @property
    def fancy_name(self) -> str:
        return "Trailing Mean Turnover, %"
