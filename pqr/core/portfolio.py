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

        longs: int | pd.DataFrame = 0 if long is None else long.values.astype(np.int8)
        shorts: int | pd.DataFrame = 0 if short is None else short.values.astype(np.int8)

        if isinstance(longs, pd.DataFrame) and isinstance(shorts, pd.DataFrame):
            longs, shorts = align(longs, shorts)

        self.picks: pd.DataFrame = array_to_alike_df_or_series(
            longs.to_numpy() - shorts.to_numpy(),
            longs
        )
        self.picks.index.name = self.name

        return self

    def weigh(self, factor: Optional[Factor] = None) -> Portfolio:
        picks, w = self.picks, self.picks
        if factor is not None:
            picks, w = [df.to_numpy() for df in align(self.picks, factor.values)]

        w_long: int = 0
        if self.has_longs():
            w_long: np.ndarray = np.where(picks == 1, w, 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                w_long: np.ndarray = np.nan_to_num(
                    w_long / np.nansum(w_long, axis=1, keepdims=True),
                    nan=0, neginf=0, posinf=0, copy=False
                )
        w_short: int = 0
        if self.has_shorts():
            w_short: np.ndarray = np.where(picks == -1, w, 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                w_short: np.ndarray = np.nan_to_num(
                    w_short / np.nansum(w_short, axis=1, keepdims=True),
                    nan=0, neginf=0, posinf=0, copy=False
                )

        if w_long == 0 and w_short == 0:
            raise ValueError("could not weigh portfolio without picks")

        self.weights: pd.DataFrame = array_to_alike_df_or_series(w_long + w_short, picks)

        return self

    def scale(
            self,
            factor: Factor,
            target: float = 1.0,
            better: Literal["better", "more"] = "more",
            leverage_limits: tuple[float, float] = (-np.inf, np.inf),
    ) -> Portfolio:
        w, factor_values = align(self.weights, factor.values)

        if better == "more":
            leverage: np.ndarray = factor_values.to_numpy() / target
        else:
            leverage: np.ndarray = target / factor_values.to_numpy()
        leverage *= w.to_numpy()
        total_leverage: np.ndarray = np.nansum(leverage, axis=1, keepdims=True)

        under_lower: int = 0
        exceed_lower: np.ndarray = total_leverage < leverage_limits[0]
        if exceed_lower.any():
            under_lower: np.ndarray = (
                    np.where(exceed_lower, leverage, 0) /
                    np.where(exceed_lower, total_leverage, 1)
            )
        above_upper: int = 0
        exceed_upper: np.ndarray = total_leverage > leverage_limits[1]
        if exceed_upper.any():
            above_upper: np.ndarray = (
                    np.where(exceed_upper, leverage, 0) /
                    np.where(exceed_upper, total_leverage, 1)
            )

        w_leveraged: np.ndarray = (
                np.where(~(exceed_lower & exceed_upper), leverage, 0) +
                under_lower +
                above_upper
        )

        self.weights: pd.DataFrame = array_to_alike_df_or_series(w_leveraged, w)

        return self

    def allocate(
            self,
            prices: pd.DataFrame,
            fee: float = 0.0,
    ) -> Portfolio:
        w, prices = align(self.weights, prices)
        positions: np.ndarray = w.to_numpy() * (1 - fee)

        universe_returns: np.ndarray = prices.pct_change().to_numpy()[1:]
        portfolio_returns: np.ndarray = (positions[:-1] * universe_returns)
        returns: np.ndarray = np.nansum(portfolio_returns, axis=1)

        self.positions: pd.DataFrame = array_to_alike_df_or_series(positions, self.weights)
        self.returns: pd.Series = array_to_alike_df_or_series(returns, self.positions.iloc[1:])

        return self

    def has_longs(self) -> bool:
        return (self.picks.to_numpy() == 1).any()

    def has_shorts(self) -> bool:
        return (self.picks.to_numpy() == -1).any()

    def __repr__(self) -> str:
        return f"Portfolio({repr(self.name)})"
