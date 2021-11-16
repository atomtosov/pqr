from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd

from .factor import Factor, Factorizer
from .portfolio import Portfolio, PortfolioBuilder
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
            portfolio_builder: PortfolioBuilder,
            add_wml: bool = False
    ):
        self.picking_strategies = picking_strategies
        self.portfolio_builder = portfolio_builder
        self.add_wml = add_wml

    def __call__(
            self,
            factor: Factor,
            universe: Universe
    ) -> list[Portfolio]:
        portfolios = [
            self.portfolio_builder(universe, longs=picker(factor), name=name)
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
                self.portfolio_builder(
                    universe,
                    longs=portfolios[0].picks,
                    shorts=portfolios[-1].picks,
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
            factorizers: dict[str, Factorizer],
            factor_model: FactorModel
    ):
        self.factorizers = factorizers
        self.factor_model = factor_model

    def __call__(
            self,
            factor: Factor,
            universe: Universe,
            target: Callable[[Portfolio], float],
    ) -> pd.DataFrame:
        metrics = []

        for index, factorizer in self.factorizers.items():
            factor = factorizer(factor.values, factor.better)

            portfolios = self.factor_model(factor, universe)

            metrics.append(
                pd.DataFrame(
                    [[target(portfolio) for portfolio in portfolios]],
                    index=index,
                    columns=[portfolio.name for portfolio in portfolios]
                )
            )

        return pd.concat(metrics)
