from __future__ import annotations

from dataclasses import dataclass, field, InitVar
from typing import Optional, Sequence, Protocol, Union

import numpy as np
import pandas as pd

from pqr.utils import align, align_many
from .factor import Factor
from .universe import Universe

__all__ = [
    "Portfolio",
    "Allocator", "AllocationStrategy",

    "EqualWeights",
    "WeightsByFactor",
    "ScalingByFactor",
    "LeverageLimits",
    "TheoreticalCommission",
    "CashAllocation",
]


class Allocator(Protocol):
    def allocate(self, positions: pd.DataFrame) -> pd.DataFrame:
        pass


AllocationStrategy = Union[Allocator, Sequence[Allocator]]


@dataclass
class Portfolio:
    longs: InitVar[Optional[pd.DataFrame]] = None
    shorts: InitVar[Optional[pd.DataFrame]] = None
    name: Optional[str] = None

    positions: Optional[pd.DataFrame] = field(init=False, repr=False)
    returns: Optional[pd.Series] = field(init=False, repr=False)

    def __post_init__(
            self,
            longs: Optional[pd.DataFrame],
            shorts: Optional[pd.DataFrame]
    ):
        if longs is None and shorts is None:
            raise ValueError("either longs or shorts must be specified")

        elif longs is not None and shorts is not None:  # long-short
            longs, shorts = align(longs, shorts)
            self.picks = pd.DataFrame(
                longs.to_numpy(dtype=np.int8) - shorts.to_numpy(dtype=np.int8),
                index=longs.index.copy(),
                columns=longs.columns.copy())
        elif longs is not None:  # long-only
            self.picks = pd.DataFrame(
                longs.to_numpy(dtype=np.int8),
                index=longs.index.copy(),
                columns=longs.columns.copy())
        else:  # short-only
            self.picks = pd.DataFrame(
                -shorts.to_numpy(dtype=np.int8),
                index=shorts.index.copy(),
                columns=shorts.columns.copy())

        if not self.name:
            self.name = "Portfolio"
        self.picks.index.name = self.name

        self.positions = None
        self.returns = None

    def allocate(self, allocation_strategy: AllocationStrategy = None) -> None:
        if allocation_strategy is None:
            allocation_strategy = [EqualWeights()]
        elif not isinstance(allocation_strategy, Sequence):
            allocation_strategy = [allocation_strategy]

        self.positions = self.picks
        for allocator in allocation_strategy:
            self.positions = allocator.allocate(self.positions)

        self.positions.index.name = self.name

    def calculate_returns(self, universe: Universe) -> None:
        prices, mask, positions = align_many(universe.prices, universe.mask, self.positions)
        universe_returns = prices.pct_change().to_numpy()[1:]

        positions_available = positions.to_numpy()[:-1]
        portfolio_returns = np.where(
            mask.to_numpy()[:-1],
            (positions_available * universe_returns), 0)

        dead_returns = np.where(
            np.isnan(portfolio_returns) & ~np.isclose(positions_available, 0),
            -positions_available, 0)
        returns = np.nansum(portfolio_returns, axis=1) + np.nansum(dead_returns, axis=1)

        self.returns = pd.Series(
            np.insert(returns, 0, 0),
            index=positions.index.copy())

        self.returns.index.name = self.name

    def get_longs(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.picks.to_numpy() == 1,
            index=self.picks.index.copy(),
            columns=self.picks.columns.copy())

    def get_shorts(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.picks.to_numpy() == -1,
            index=self.picks.index.copy(),
            columns=self.picks.columns.copy())


@dataclass
class EqualWeights:
    leverage: float = 1.0

    def allocate(self, positions: pd.DataFrame) -> pd.DataFrame:
        picks_array = positions.to_numpy()
        longs, shorts = picks_array > 0, picks_array < 0

        with np.errstate(divide="ignore", invalid="ignore"):
            return pd.DataFrame(
                (normalize(longs) - normalize(shorts)) * self.leverage,
                index=positions.index.copy(),
                columns=positions.columns.copy())


@dataclass
class WeightsByFactor:
    factor: Factor
    leverage: float = 1.0

    def allocate(self, positions: pd.DataFrame) -> pd.DataFrame:
        positions, factor_values = align(positions, self.factor.values)
        picks_array, factor_array = positions.to_numpy(), factor_values.to_numpy()
        longs, shorts = picks_array > 0, picks_array < 0

        with np.errstate(divide="ignore", invalid="ignore"):
            return pd.DataFrame(
                (normalize(longs * factor_array) - normalize(shorts * factor_array)) * self.leverage,
                index=positions.index.copy(),
                columns=positions.columns.copy())


@dataclass
class ScalingByFactor:
    factor: Factor
    target: float = 1.0

    def allocate(self, positions: pd.DataFrame) -> pd.DataFrame:
        positions, factor_values = align(positions, self.factor.values)

        if self.factor.is_better_more():
            leverage = factor_values.to_numpy() / self.target
        else:
            leverage = self.target / factor_values.to_numpy()

        return pd.DataFrame(
            leverage * positions.to_numpy(),
            index=positions.index.copy(),
            columns=positions.columns.copy())


@dataclass
class LeverageLimits:
    min_leverage: float = -np.inf
    max_leverage: float = np.inf

    def allocate(self, positions: pd.DataFrame) -> pd.DataFrame:
        w = positions.to_numpy()
        total_leverage = np.nansum(w, axis=1, keepdims=True)

        with np.errstate(divide="ignore", invalid="ignore"):
            exceed_min = total_leverage < self.min_leverage
            under_min = (np.where(exceed_min, w, 0) /
                         np.where(exceed_min, total_leverage, 1)) * self.min_leverage

            exceed_max = total_leverage > self.max_leverage
            above_max = (np.where(exceed_max, w, 0) /
                         np.where(exceed_max, total_leverage, 1)) * self.max_leverage

        return pd.DataFrame(
            np.where(~(exceed_min | exceed_max), w, 0) +
            under_min + above_max,
            index=positions.index.copy(),
            columns=positions.columns.copy())


@dataclass
class TheoreticalCommission:
    fee: float = 0.0

    def allocate(self, positions: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            positions.to_numpy() * (1 - self.fee),
            index=positions.index.copy(),
            columns=positions.columns.copy())


@dataclass
class CashAllocation:
    prices: pd.DataFrame = field(repr=False)
    capital: float = 1_000_000.0
    fee: float = 0.0

    def allocate(self, positions: pd.DataFrame) -> pd.DataFrame:
        prices, weights = align(self.prices, positions)
        prices_arr = prices.to_numpy()
        weights_arr = weights.to_numpy()

        allocation = np.zeros_like(weights_arr, dtype=int)
        cash = np.ones(allocation.shape[0]) * self.capital
        balance = cash.copy()

        for i in range(len(allocation)):
            w, p = weights_arr[i], prices_arr[i]
            prev_alloc = allocation[max(0, i - 1)]
            prev_cash = cash[max(0, i - 1)]

            allocation[i], cash[i], balance[i] = self._allocation_step(
                w, p, prev_alloc, prev_cash)

        positions_in_cash = np.nan_to_num(
            np.insert(allocation * prices_arr,
                      obj=0, axis=1, values=cash),
            nan=0, neginf=0, posinf=0
        )

        return pd.DataFrame(
            positions_in_cash / balance[:, np.newaxis],
            index=weights.index.copy(),
            columns=["CASH_RESIDUALS"] + list(weights.columns.copy()))

    def _allocation_step(
            self,
            weights: np.ndarray,
            prices: np.ndarray,
            prev_alloc: np.ndarray,
            prev_cash: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        balance = prev_cash + np.nansum(prev_alloc * prices)

        allocation_delta = -prev_alloc * (weights == 0)
        commission = self._accrue_commission(allocation_delta, prices)

        to_rebalance = (allocation_delta == 0) & (prev_alloc != 0) & (weights != 0)
        allocation_delta[to_rebalance] = self._rebalance_positions(
            weights[to_rebalance],
            prices[to_rebalance],
            prev_alloc[to_rebalance],
            balance - commission
        )
        commission = self._accrue_commission(allocation_delta, prices)

        to_open = (allocation_delta == 0) & (prev_alloc == 0) & (weights != 0)
        allocation_delta[to_open] = self._open_positions(
            weights[to_open],
            prices[to_open],
            prev_alloc[to_open],
            balance - commission
        )

        cash_delta = -(allocation_delta * prices)
        commission = self._accrue_commission(allocation_delta, prices)

        allocation_delta = prev_alloc + allocation_delta
        cash = prev_cash + np.nansum(cash_delta) - commission

        return allocation_delta, cash, balance

    def _close_positions(
            self,
            weights: np.ndarray,
            prev_alloc: np.ndarray
    ) -> np.ndarray:
        return -prev_alloc * (weights == 0)

    def _rebalance_positions(
            self,
            weights: np.ndarray,
            prices: np.ndarray,
            prev_alloc: np.ndarray,
            balance: float
    ) -> np.ndarray:
        ideal_alloc = balance * weights / prices
        allocation = (ideal_alloc - prev_alloc)
        allocation[allocation > 0] *= 1 - self.fee
        allocation[allocation < 0] *= 1 + self.fee

        buy, sell = allocation > 0, allocation < 0
        long, short = prev_alloc > 0, prev_alloc < 0
        buy_more, sell_some = buy & long, sell & long  # rebalance longs
        sell_more, buy_some = sell & short, buy & short  # rebalance shorts

        allocation[buy_more | sell_more] = np.trunc(allocation[buy_more | sell_more])
        allocation[sell_some] = np.floor(allocation[sell_some])
        allocation[buy_some] = np.ceil(allocation[buy_some])

        return np.nan_to_num(allocation, nan=0)

    def _open_positions(
            self,
            weights: np.ndarray,
            prices: np.ndarray,
            prev_alloc: np.ndarray,
            balance: float
    ) -> np.ndarray:
        ideal_alloc = balance * weights / prices
        allocation = (ideal_alloc - prev_alloc) * (1 - self.fee)
        allocation[allocation > 0] *= 1 - self.fee
        allocation[allocation < 0] *= 1 + self.fee

        return np.nan_to_num(np.trunc(allocation), nan=0)

    def _accrue_commission(
            self,
            allocation: np.ndarray,
            prices: np.ndarray,
    ) -> float:
        return np.nansum(np.abs(allocation * prices)) * self.fee


def normalize(raw_weights: np.ndarray) -> np.ndarray:
    normalizing_coef = np.nansum(raw_weights, axis=1, keepdims=True, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.nan_to_num(raw_weights / normalizing_coef,
                             nan=0, neginf=0, posinf=0)
