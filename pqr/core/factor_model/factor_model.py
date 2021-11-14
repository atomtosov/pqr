from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import pandas as pd

from pqr.core.factor import Factor
from pqr.core.portfolio import Portfolio, PortfolioBuilder

__all__ = [
    "FactorModel",
]

PickingStrategy = Callable[[Factor], pd.DataFrame]


@dataclass
class FactorModel:
    picking_strategies: Sequence[PickingStrategy]
    portfolio_builder: PortfolioBuilder
    add_wml: bool = False

    def __call__(self, factor: Factor) -> list[Portfolio]:
        portfolios = [
            self.portfolio_builder(longs=picker(factor), name=name)
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
                    longs=portfolios[0].picks,
                    shorts=portfolios[-1].picks,
                    name="WML"
                )
            )

        return portfolios
