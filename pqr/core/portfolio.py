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
    "AllocationStep",

    "EqualWeights",
    "WeightsByFactor",
    "ScalingByFactor",
    "LeverageLimits",
    "TheoreticalAllocation",
    "CashAllocation",
]

AllocationStep = Callable[[pd.DataFrame], pd.DataFrame]


class Portfolio:
    def __init__(
            self,
            universe: Universe,
            longs: Optional[pd.DataFrame] = None,
            shorts: Optional[pd.DataFrame] = None,
            allocation_strategy: Optional[AllocationStep | Sequence[AllocationStep]] = None,
            name: Optional[str] = None
    ):
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
        self.picks = picks

        if allocation_strategy is None:
            allocation_strategy = EqualWeights()
        elif isinstance(allocation_strategy, Sequence):
            allocation_strategy = compose(*allocation_strategy)
        self.positions = allocation_strategy(self.picks)

        self.returns = universe(self.positions)

        if name is None:
            name = "Portfolio"
        self.name = name

        self.picks.index.name = name
        self.positions.index.name = name
        self.returns.index.name = name

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

    def __call__(self, weights: pd.DataFrame) -> pd.DataFrame:
        prices, weights = align(self.prices, weights)
        prices_arr = prices.to_numpy()
        weights_arr = weights.to_numpy()

        allocation = np.zeros_like(weights_arr, dtype=float)
        cash = np.ones(allocation.shape[0]) * self.capital
        balance = cash.copy()

        for i in range(len(allocation)):
            w, p, c = weights_arr[i], prices_arr[i], cash[i]
            prev_alloc = allocation[max(0, i - 1)]

            balance[i] = c + np.nansum(prev_alloc * p)

            alloc = np.nan_to_num(w * balance[i] / p).astype(int)
            alloc_diff = alloc - prev_alloc
            cash_diff = -(alloc_diff * p)
            commission = np.nansum(np.abs(cash_diff)) * self.fee

            cash[i] = c + np.nansum(cash_diff) - commission
            allocation[i] = alloc

        positions_in_cash = np.insert(
            allocation * prices_arr,
            obj=0, axis=1, values=cash,
        )

        return pd.DataFrame(
            positions_in_cash / balance[:, np.newaxis],
            index=weights.index.copy(),
            columns=["CASH_RESIDUALS"] + list(weights.columns.copy())
        )
