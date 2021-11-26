from __future__ import annotations

from typing import Optional, Callable, Sequence

import numpy as np
import pandas as pd

from pqr.utils import align
from .factor import Factor
from .universe import Universe
from .utils import compose, normalize

__all__ = [
    "Portfolio",
    "PortfolioBuilder",

    "EqualWeights",
    "WeightsByFactor",
    "ScalingByFactor",
    "LeverageLimits",

    "TheoreticalAllocation",
    "CashAllocation",
]

BuildingStep = Callable[[pd.DataFrame], pd.DataFrame]


class Portfolio:
    def __init__(
            self,
            picks: pd.DataFrame,
            weights: pd.DataFrame,
            positions: pd.DataFrame,
            returns: pd.Series,
            name: Optional[str] = None
    ):
        self.picks = picks.astype(np.int8)
        self.weights = weights.astype(float)
        self.positions = positions.astype(float)
        self.returns = returns.astype(float)

        self.name = name if name is not None else "Portfolio"
        self.picks.index.name = self.name
        self.weights.index.name = self.name
        self.positions.index.name = self.name
        self.returns.index.name = self.name

    def get_longs(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.picks.to_numpy() == 1,
            index=self.picks.index.copy(),
            columns=self.picks.columns.copy()
        )

    def get_shorts(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.picks.to_numpy() == -1,
            index=self.picks.index.copy(),
            columns=self.picks.columns.copy()
        )


class PortfolioBuilder:
    def __init__(
            self,
            weighting_strategy: BuildingStep | Sequence[BuildingStep],
            allocation_strategy: BuildingStep | Sequence[BuildingStep],
    ):
        if isinstance(weighting_strategy, Sequence):
            self.weighting_strategy = compose(*weighting_strategy)
        else:
            self.weighting_strategy = weighting_strategy

        if isinstance(allocation_strategy, Sequence):
            self.allocation_strategy = compose(*allocation_strategy)
        else:
            self.allocation_strategy = allocation_strategy

    def __call__(
            self,
            universe: Universe,
            longs: Optional[pd.DataFrame] = None,
            shorts: Optional[pd.DataFrame] = None,
            name: Optional[str] = None
    ) -> Portfolio:
        if longs is None and shorts is None:
            raise ValueError("either longs or shorts must be specified")

        elif longs is not None and shorts is not None:  # long-short
            longs, shorts = align(longs, shorts)
            picks = pd.DataFrame(
                longs.to_numpy(dtype=np.int8) - shorts.to_numpy(dtype=np.int8),
                index=longs.index.copy(),
                columns=longs.columns.copy()
            )
        elif longs is not None:  # long-only
            picks = pd.DataFrame(
                longs.to_numpy(dtype=np.int8),
                index=longs.index.copy(),
                columns=longs.columns.copy()
            )
        else:  # short-only
            picks = pd.DataFrame(
                -shorts.to_numpy(dtype=np.int8),
                index=shorts.index.copy(),
                columns=shorts.columns.copy()
            )

        weights = self.weighting_strategy(picks)
        positions = self.allocation_strategy(weights)
        returns = universe(positions)

        return Portfolio(
            picks,
            weights,
            positions,
            returns,
            name=name
        )


class EqualWeights:
    def __call__(self, picks: pd.DataFrame) -> pd.DataFrame:
        picks_array = picks.to_numpy()
        longs, shorts = picks_array == 1, picks_array == -1

        if not longs.any() and not shorts.any():
            raise ValueError("cannot weigh portfolio without picks")

        with np.errstate(divide="ignore", invalid="ignore"):
            return pd.DataFrame(
                normalize(longs) - normalize(shorts),
                index=picks.index.copy(),
                columns=picks.columns.copy()
            )


class WeightsByFactor:
    def __init__(self, factor: Factor):
        self.factor = factor

    def __call__(self, picks: pd.DataFrame) -> pd.DataFrame:
        picks, factor_values = align(picks, self.factor.values)
        picks_array, factor_array = picks.to_numpy(), factor_values.to_numpy()
        longs, shorts = picks_array == 1, picks_array == -1

        if not longs.any() and not shorts.any():
            raise ValueError("cannot weigh portfolio without picks")

        with np.errstate(divide="ignore", invalid="ignore"):
            return pd.DataFrame(
                normalize(longs * factor_array) - normalize(shorts * factor_array),
                index=picks.index.copy(),
                columns=picks.columns.copy()
            )


class ScalingByFactor:
    def __init__(
            self,
            factor: Factor,
            target: float = 1.0
    ):
        self.factor = factor
        self.target = target

    def __call__(self, weights: pd.DataFrame) -> pd.DataFrame:
        weights, factor_values = align(weights, self.factor.values)

        if self.factor.is_better_more():
            leverage = factor_values.to_numpy() / self.target
        else:
            leverage = self.target / factor_values.to_numpy()

        return pd.DataFrame(
            leverage * weights.to_numpy(),
            index=weights.index.copy(),
            columns=weights.columns.copy()
        )


class LeverageLimits:
    def __init__(
            self,
            min_leverage: float = -np.inf,
            max_leverage: float = np.inf
    ):
        self.min_leverage = min_leverage
        self.max_leverage = max_leverage

    def __call__(self, weights: pd.DataFrame) -> pd.DataFrame:
        w = weights.to_numpy()
        total_leverage = np.nansum(w, axis=1, keepdims=True)

        with np.errstate(divide="ignore", invalid="ignore"):
            exceed_min = total_leverage < self.min_leverage
            under_min = (
                                np.where(exceed_min, w, 0) /
                                np.where(exceed_min, total_leverage, 1)
                        ) * self.min_leverage

            exceed_max = total_leverage > self.max_leverage
            above_max = (
                                np.where(exceed_max, w, 0) /
                                np.where(exceed_max, total_leverage, 1)
                        ) * self.max_leverage

        return pd.DataFrame(
            np.where(~(exceed_min | exceed_max), w, 0) +
            under_min + above_max,
            index=weights.index.copy(),
            columns=weights.columns.copy()
        )


class TheoreticalAllocation:
    def __init__(self, fee: float = 0.0):
        self.fee = fee

    def __call__(self, weights: pd.DataFrame) -> pd.DataFrame:
        w = weights.to_numpy()
        positions = pd.DataFrame(
            w * (1 - self.fee),
            index=weights.index.copy(),
            columns=weights.columns.copy()
        )
        positions.insert(
            loc=0,
            column="CASH_RESIDUALS",
            value=(1 - self.fee) - np.nansum(positions, axis=1)
        )

        return positions


class CashAllocation:
    def __init__(
            self,
            prices: pd.DataFrame,
            capital: float = 1_000_000.0,
            fee: float = 0.0
    ):
        self.prices = prices
        self.capital = capital
        self.fee = fee

    def __call__(self, portfolio: Portfolio) -> Portfolio:
        prices, weights = align(self.prices, portfolio.weights)
        prices_arr = prices.to_numpy()
        weights_arr = weights.to_numpy()

        allocation = np.zeros_like(weights_arr, dtype=float)
        cash = np.ones(allocation.shape[0]) * self.capital
        balance = cash.copy()

        for i in range(len(allocation)):
            w, p, c = weights_arr[i], prices_arr[i], cash[i]
            prev_alloc = allocation[min(0, i - 1)]

            balance[i] = c + np.nansum(prev_alloc * p)

            ideal_alloc = np.nan_to_num(w * balance[i]).astype(int)
            ideal_alloc_diff = ideal_alloc - prev_alloc
            ideal_commission = np.nansum(np.abs(ideal_alloc_diff * p)) * self.fee
            max_allowed = balance[i] - ideal_commission

            alloc = np.nan_to_num(w * max_allowed / p).astype(int)
            alloc_diff = alloc - prev_alloc
            cash_diff = -(alloc_diff * p)
            commission = np.nansum(np.abs(cash_diff)) * self.fee

            cash[i] = c + np.nansum(cash_diff) - commission
            allocation[i] = alloc

        positions_in_cash = np.insert(
            allocation * prices_arr,
            obj=0, axis=1,
            values=cash,
        )

        portfolio.set_positions(
            pd.DataFrame(
                positions_in_cash / balance[:, np.newaxis],
                index=weights.index.copy(),
                columns=["CASH_RESIDUALS"] + list(weights.columns.copy())
            )
        )

        return portfolio
