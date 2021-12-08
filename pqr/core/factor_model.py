from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Protocol

import numpy as np
import pandas as pd

from .factor import Factor, Preprocessor
from .portfolio import Portfolio, Allocator
from .universe import Universe

__all__ = [
    "FactorModel",

    "Quantiles", "split_quantiles",
    "Top",
    "Bottom",
    "TimeSeries",

    "PreprocessorsGrid",
]


class PickingStrategy(Protocol):
    def pick(self, factor: Factor) -> pd.DataFrame:
        pass


@dataclass
class FactorModel:
    picking_strategies: Sequence[PickingStrategy]
    allocation_strategy: Allocator | Sequence[Allocator]
    add_wml: bool = False

    def __call__(
            self,
            factor: Factor,
            universe: Universe
    ) -> list[Portfolio]:
        portfolios_names = [
            "Winners",
            *[f"Neutral {i}" for i in range(1, len(self.picking_strategies) - 1)],
            "Losers"
        ]
        portfolios = []
        for picking_strategy, name in zip(self.picking_strategies, portfolios_names):
            portfolio = Portfolio(
                longs=picking_strategy.pick(factor),
                name=name
            )
            portfolio.allocate(self.allocation_strategy)
            portfolio.calculate_returns(universe)

            portfolios.append(portfolio)

        if self.add_wml:
            wml = Portfolio(
                longs=portfolios[0].picks.astype(bool),
                shorts=portfolios[-1].picks.astype(bool),
                name="WML"
            )
            wml.allocate(self.allocation_strategy)
            wml.calculate_returns(universe)

            portfolios.append(wml)

        return portfolios


@dataclass
class Quantiles:
    min_q: float = 0.0
    max_q: float = 0.0

    def pick(self, factor: Factor) -> pd.DataFrame:
        factor_values = factor.values.to_numpy()
        q = factor.quantile([self.min_q, self.max_q]).to_numpy()

        return pd.DataFrame(
            (q[:, [1]] <= factor_values) & (factor_values <= q[:, [0]])
            if factor.is_better_more() else
            (q[:, [0]] <= factor_values) & (factor_values <= q[:, [1]]),
            index=factor.values.index.copy(),
            columns=factor.values.columns.copy()
        )


@dataclass
class Top:
    n: int = 10

    def pick(self, factor: Factor) -> pd.DataFrame:
        factor_values = factor.values.to_numpy()
        top = factor.top(self.n).to_numpy()

        return pd.DataFrame(
            factor_values >= top if factor.is_better_more() else factor_values <= top,
            index=factor.values.index.copy(),
            columns=factor.values.columns.copy()
        )


@dataclass
class Bottom:
    n: int = 10

    def pick(self, factor: Factor) -> pd.DataFrame:
        factor_values = factor.values.to_numpy()
        bottom = factor.bottom(self.n).to_numpy()

        return pd.DataFrame(
            factor_values <= bottom if factor.is_better_more() else factor_values >= bottom,
            index=factor.values.index.copy(),
            columns=factor.values.columns.copy()
        )


@dataclass
class TimeSeries:
    min_threshold: float = -np.inf
    max_threshold: float = np.inf

    def pick(self, factor: Factor) -> pd.DataFrame:
        factor_values = factor.values.to_numpy()

        return pd.DataFrame(
            (self.min_threshold <= factor_values) & (factor_values <= self.max_threshold),
            index=factor.values.index.copy(),
            columns=factor.values.columns.copy()
        )


@dataclass
class PreprocessorsGrid:
    factor_preprocessors: dict[str, Preprocessor | Sequence[Preprocessor]]
    factor_model: FactorModel

    def search(
            self,
            factor: Factor,
            universe: Universe,
            target: Callable[[Portfolio], float],
    ) -> pd.DataFrame:
        metrics = []

        for name, factor_preprocessor in self.factor_preprocessors.items():
            portfolios = self.factor_model(
                Factor(factor.values, factor.better, factor_preprocessor),
                universe
            )

            metrics.append(
                pd.Series(
                    {
                        portfolio.name: target(portfolio)
                        for portfolio in portfolios
                    },
                    name=name
                )
            )

        return pd.DataFrame(metrics)


def split_quantiles(n: int) -> list[Quantiles]:
    q = np.linspace(0, 1, n + 1)
    return [
        Quantiles(q[i], q[i + 1])
        for i in range(len(q) - 1)
    ]
