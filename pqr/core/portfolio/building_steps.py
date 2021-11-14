from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pqr.core.factor import Factor
from pqr.core.utils import normalize
from pqr.utils import align
from .portfolio import Portfolio

__all__ = [
    "EqualWeights",
    "WeightsByFactor",
    "RelativePositions",
]


@dataclass
class EqualWeights:
    def __call__(self, portfolio: Portfolio) -> Portfolio:
        longs, shorts = portfolio.get_longs(), portfolio.get_shorts()
        longs_any, shorts_any = longs.to_numpy().any(), shorts.to_numpy().any()

        if not longs_any and not shorts_any:
            raise ValueError("cannot weigh portfolio without picks")

        elif longs_any and shorts_any:
            weights = normalize(longs) - normalize(shorts)
        elif longs_any:
            weights = normalize(longs)
        else:
            weights = -normalize(shorts)

        portfolio.set_weights(weights)

        return portfolio


@dataclass
class WeightsByFactor:
    factor: Factor

    def __call__(self, portfolio: Portfolio) -> Portfolio:
        longs, shorts = portfolio.get_longs(), portfolio.get_shorts()
        longs_any, shorts_any = longs.to_numpy().any(), shorts.to_numpy().any()

        if not longs_any and not shorts_any:
            raise ValueError("cannot weigh portfolio without picks")

        elif longs_any and shorts_any:
            factor_values, longs = align(self.factor.values, longs)
            weights = normalize(longs * factor_values) - normalize(shorts * factor_values)
        elif longs_any:
            factor_values, longs = align(self.factor.values, longs)
            weights = normalize(longs * longs)
        else:
            factor_values, shorts = align(self.factor.values, shorts)
            weights = -normalize(shorts * factor_values)

        portfolio.set_weights(weights)

        return portfolio


@dataclass
class ScalingByFactor:
    factor: Factor
    target: float = 1.0

    def __call__(self, portfolio: Portfolio) -> Portfolio:
        w, factor_values = align(portfolio.weights, self.factor.values)

        if self.factor.is_better_more():
            leverage = factor_values.to_numpy() / self.target
        else:
            leverage = self.target / factor_values.to_numpy()
        leveraged_weights = leverage * w.to_numpy()

        portfolio.set_weights(
            pd.DataFrame(
                leveraged_weights,
                index=w.index.copy(),
                columns=w.columns.copy()
            )
        )

        return portfolio


@dataclass
class LeverageLimits:
    min_leverage: float = -np.inf
    max_leverage: float = np.inf

    def __call__(self, portfolio: Portfolio) -> Portfolio:
        w = portfolio.weights.to_numpy()
        total_leverage = np.nansum(w, axis=1, keepdims=True)

        exceed_min = total_leverage < self.min_leverage
        if exceed_min.any():
            under_min = (
                    np.where(exceed_min, w, 0) /
                    np.where(exceed_min, total_leverage, 1)
            )
        else:
            under_min = 0

        exceed_max = total_leverage > self.max_leverage
        if exceed_max.any():
            above_max = (
                    np.where(exceed_max, w, 0) /
                    np.where(exceed_max, total_leverage, 1)
            )
        else:
            above_max = 0

        portfolio.set_weights(
            pd.DataFrame(
                np.where(~(exceed_min & exceed_max), w, 0) +
                under_min + above_max,
                index=portfolio.weights.index.copy(),
                columns=portfolio.weights.columns.copy()
            )
        )

        return portfolio


@dataclass
class RelativePositions:
    prices: pd.DataFrame = field(repr=False)
    fee: float = 0.0

    def __call__(self, portfolio: Portfolio) -> Portfolio:
        w, prices = align(portfolio.weights, self.prices)
        positions = w.to_numpy() * (1 - self.fee)

        universe_returns = prices.pct_change().to_numpy()[1:]
        portfolio_returns = (positions[:-1] * universe_returns)
        dead_returns = np.where(
            np.isnan(portfolio_returns) & ~np.isclose(positions[:-1], 0),
            -positions[:-1], 0
        )
        returns = np.nansum(portfolio_returns, axis=1) + np.nansum(dead_returns, axis=1)

        portfolio.set_positions(
            pd.DataFrame(
                positions,
                index=w.index.copy(),
                columns=w.columns.copy()
            )
        )
        portfolio.set_returns(
            pd.Series(
                returns,
                index=portfolio.positions.index[1:].copy()
            )
        )

        return portfolio
