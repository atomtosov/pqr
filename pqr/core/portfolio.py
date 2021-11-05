from __future__ import annotations

from typing import Optional, Literal

import numpy as np
import pandas as pd

from .factor import Factor
from .universe import Universe
from ..utils import align, array_to_alike_df_or_series

__all__ = [
    "Portfolio",
]


class Portfolio:
    __slots__ = (
        "name",
        "picks",
        "weights",
        "positions",
        "returns",
    )

    def __init__(self, name: str = "portfolio"):
        self.name = name

        self.picks: pd.DataFrame = pd.DataFrame(dtype=np.int8)
        self.weights: pd.DataFrame = pd.DataFrame(dtype=float)
        self.positions: pd.DataFrame = pd.DataFrame(dtype=float)
        self.returns: pd.Series = pd.Series(dtype=float)

    def pick(
            self,
            long: Optional[Universe] = None,
            short: Optional[Universe] = None,
    ) -> Portfolio:
        if long is None and short is None:
            raise ValueError("either long or short must be specified")
        elif long is not None and short is not None:  # long-short
            longs, shorts = align(long.values, short.values)
            longs, shorts = longs.to_numpy(dtype=np.int8), shorts.to_numpy(dtype=np.int8)
        elif long is not None:  # long-only
            longs = long.values.to_numpy(dtype=np.int8)
            shorts = 0
        else:  # short-only
            longs = 0
            shorts = short.values.to_numpy(dtype=np.int8)

        self.picks = array_to_alike_df_or_series(
            longs - shorts,
            long.values
        )
        self.picks.index.name = self.name

        return self

    def weigh(self, factor: Optional[Factor] = None) -> Portfolio:
        picks, w = self.picks, self.picks
        if factor is not None:
            picks, w = [df.to_numpy() for df in align(self.picks, factor.values)]

        w_long = 0
        if self.has_longs():
            w_long = np.where(picks == 1, w, 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                w_long = np.nan_to_num(
                    w_long / np.nansum(w_long, axis=1, keepdims=True),
                    nan=0, neginf=0, posinf=0, copy=False
                )
        w_short = 0
        if self.has_shorts():
            w_short = np.where(picks == -1, w, 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                w_short = np.nan_to_num(
                    w_short / np.nansum(w_short, axis=1, keepdims=True),
                    nan=0, neginf=0, posinf=0, copy=False
                )

        if isinstance(w_long, int) and isinstance(w_short, int):
            raise ValueError("could not weigh portfolio without picks")

        self.weights = array_to_alike_df_or_series(w_long - w_short, picks)

        return self

    def scale(
            self,
            factor: Factor,
            target: float = 1.0,
            better: Literal["more", "less"] = "more",
            leverage_limits: tuple[float, float] = (-np.inf, np.inf),
    ) -> Portfolio:
        w, factor_values = align(self.weights, factor.values)

        if better == "more":
            leverage = factor_values.to_numpy() / target
        else:
            leverage = target / factor_values.to_numpy()
        leverage *= w.to_numpy()
        total_leverage = np.nansum(leverage, axis=1, keepdims=True)

        under_lower = 0
        exceed_lower = total_leverage < leverage_limits[0]
        if exceed_lower.any():
            under_lower = (
                    np.where(exceed_lower, leverage, 0) /
                    np.where(exceed_lower, total_leverage, 1)
            )
        above_upper = 0
        exceed_upper = total_leverage > leverage_limits[1]
        if exceed_upper.any():
            above_upper = (
                    np.where(exceed_upper, leverage, 0) /
                    np.where(exceed_upper, total_leverage, 1)
            )

        w_leveraged = (
                np.where(~(exceed_lower & exceed_upper), leverage, 0) +
                under_lower +
                above_upper
        )

        self.weights = array_to_alike_df_or_series(w_leveraged, w)

        return self

    def allocate(
            self,
            prices: pd.DataFrame,
            fee: float = 0.0,
    ) -> Portfolio:
        w, prices = align(self.weights, prices)
        positions = w.to_numpy() * (1 - fee)

        universe_returns = prices.pct_change().to_numpy()[1:]
        portfolio_returns = (positions[:-1] * universe_returns)
        dead_returns = np.where(
            np.isnan(portfolio_returns) & ~np.isclose(positions[:-1], 0),
            -positions[:-1], 0
        )
        returns = np.nansum(portfolio_returns, axis=1) + np.nansum(dead_returns, axis=1)

        self.positions = array_to_alike_df_or_series(positions, prices)
        self.returns = array_to_alike_df_or_series(returns, self.positions.iloc[1:])

        return self

    def has_longs(self) -> bool:
        return (self.picks.to_numpy() == 1).any()

    def has_shorts(self) -> bool:
        return (self.picks.to_numpy() == -1).any()

    def __repr__(self) -> str:
        return f"Portfolio({repr(self.name)})"
