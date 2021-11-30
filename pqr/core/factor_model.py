from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd

from .factor import Factor, Preprocessor
from .portfolio import Portfolio, AllocationStep
from .universe import Universe

__all__ = [
    "FactorModel",

    "Quantiles",
    "Top",
    "Bottom",
    "TimeSeries",

    "GridSearch",
]


class FactorModel:
    def __init__(
            self,
            picking_strategies: Sequence[Callable[[Factor], pd.DataFrame]],
            allocation_strategy: AllocationStep | Sequence[AllocationStep],
            add_wml: bool = False
    ):
        self.picking_strategies = picking_strategies
        self.allocation_strategy = allocation_strategy
        self.add_wml = add_wml

    def __call__(
            self,
            factor: Factor,
            universe: Universe
    ) -> list[Portfolio]:
        portfolios = [
            Portfolio(
                universe,
                longs=picker(factor),
                allocation_strategy=self.allocation_strategy,
                name=name
            )
            for picker, name in zip(
                self.picking_strategies,
                [
                    "Winners",
                    *[f"Neutral {i}" for i in range(1, len(self.picking_strategies) - 1)],
                    "Losers"
                ]
            )
        ]

        if self.add_wml:
            portfolios.append(
                Portfolio(
                    universe,
                    longs=portfolios[0].picks.astype(bool),
                    shorts=portfolios[-1].picks.astype(bool),
                    allocation_strategy=self.allocation_strategy,
                    name="WML"
                )
            )

        return portfolios


class Quantiles:
    def __init__(
            self,
            min_q: float = 0.0,
            max_q: float = 1.0
    ):
        self.min_q = min_q
        self.max_q = max_q

    def __call__(self, factor: Factor) -> pd.DataFrame:
        factor_values = factor.values.to_numpy()
        q = factor.quantile([self.min_q, self.max_q]).to_numpy()

        return pd.DataFrame(
            (q[:, [1]] <= factor_values) & (factor_values <= q[:, [0]])
            if factor.is_better_more() else
            (q[:, [0]] <= factor_values) & (factor_values <= q[:, [1]]),
            index=factor.values.index.copy(),
            columns=factor.values.columns.copy()
        )


class Top:
    def __init__(self, n: int = 10):
        self.n = n

    def __call__(self, factor: Factor) -> pd.DataFrame:
        factor_values = factor.values.to_numpy()
        top = factor.top(self.n).to_numpy()

        return pd.DataFrame(
            factor_values >= top if factor.is_better_more() else factor_values <= top,
            index=factor.values.index.copy(),
            columns=factor.values.columns.copy()
        )


class Bottom:
    def __init__(self, n: int = 10):
        self.n = n

    def __call__(self, factor: Factor) -> pd.DataFrame:
        factor_values = factor.values.to_numpy()
        bottom = factor.bottom(self.n).to_numpy()

        return pd.DataFrame(
            factor_values <= bottom if factor.is_better_more() else factor_values >= bottom,
            index=factor.values.index.copy(),
            columns=factor.values.columns.copy()
        )


class TimeSeries:
    def __init__(
            self,
            min_threshold: float = -np.inf,
            max_threshold: float = np.inf
    ):
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    def __call__(self, factor: Factor) -> pd.DataFrame:
        factor_values = factor.values.to_numpy()

        return pd.DataFrame(
            (self.min_threshold <= factor_values) & (factor_values <= self.max_threshold),
            index=factor.values.index.copy(),
            columns=factor.values.columns.copy()
        )


class GridSearch:
    def __init__(
            self,
            factor_preprocessors: dict[str, Preprocessor | Sequence[Preprocessor]],
            factor_model: FactorModel
    ):
        self.factor_preprocessors = factor_preprocessors
        self.factor_model = factor_model

    def __call__(
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
    quantiles = [
        Quantiles(q[i], q[i + 1])
        for i in range(len(q) - 1)
    ]

    return quantiles
