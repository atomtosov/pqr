from typing import Optional, Callable

import numpy as np
import pandas as pd

from pqr.utils import align, compose, normalize
from .factor import Factor
from .universe import Universe

__all__ = [
    "Portfolio",
    "PortfolioBuilder",

    "EqualWeights",
    "WeightsByFactor",
    "ScalingByFactor",
    "LeverageLimits",

    "TheoreticalAllocation",
    "CashAllocation",

    "CalculateReturns",
]


class Portfolio:
    def __init__(
            self,
            picks: pd.DataFrame,
            name: Optional[str] = None
    ):
        self.picks = picks.astype(np.int8)
        self.name = name if name is not None else "Portfolio"
        self.picks.index.name = self.name

        self.weights = pd.DataFrame(dtype=float)
        self.positions = pd.DataFrame(dtype=float)
        self.returns = pd.Series(dtype=float)

    def set_picks(self, picks: pd.DataFrame) -> None:
        self.picks = picks
        self.picks.index.name = self.name

    def set_weights(self, weights: pd.DataFrame) -> None:
        self.weights = weights
        self.weights.index.name = self.name

    def set_positions(self, positions: pd.DataFrame) -> None:
        self.positions = positions
        self.positions.index.name = self.name

    def set_returns(self, returns: pd.Series) -> None:
        self.returns = returns
        self.returns.name = self.name

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
    def __init__(self, *building_steps: Callable[[Portfolio], Portfolio]):
        self._builder = compose(*building_steps)

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

        portfolio = self._builder(Portfolio(picks, name))

        return CalculateReturns(universe)(portfolio)


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


class WeightsByFactor:
    def __init__(self, factor: Factor):
        self.factor = factor

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
            weights = normalize(longs * factor_values)
        else:
            factor_values, shorts = align(self.factor.values, shorts)
            weights = -normalize(shorts * factor_values)

        portfolio.set_weights(weights)

        return portfolio


class ScalingByFactor:
    def __init__(
            self,
            factor: Factor,
            target: float = 1.0
    ):
        self.factor = factor
        self.target = target

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


class LeverageLimits:
    def __init__(
            self,
            min_leverage: float = -np.inf,
            max_leverage: float = np.inf
    ):
        self.min_leverage = min_leverage
        self.max_leverage = max_leverage

    def __call__(self, portfolio: Portfolio) -> Portfolio:
        w = portfolio.weights.to_numpy()
        total_leverage = np.nansum(w, axis=1, keepdims=True)

        exceed_min = total_leverage < self.min_leverage
        if exceed_min.any():
            under_min = (
                    np.where(exceed_min, w, 0) /
                    np.where(exceed_min, total_leverage, 1)
            ) * self.min_leverage
        else:
            under_min = 0

        exceed_max = total_leverage > self.max_leverage
        if exceed_max.any():
            above_max = (
                    np.where(exceed_max, w, 0) /
                    np.where(exceed_max, total_leverage, 1)
            ) * self.max_leverage
        else:
            above_max = 0

        portfolio.set_weights(
            pd.DataFrame(
                np.where(~(exceed_min | exceed_max), w, 0) +
                under_min + above_max,
                index=portfolio.weights.index.copy(),
                columns=portfolio.weights.columns.copy()
            )
        )

        return portfolio


class TheoreticalAllocation:
    def __init__(self, fee: float = 0.0):
        self.fee = fee

    def __call__(self, portfolio: Portfolio) -> Portfolio:
        positions = pd.DataFrame(
            portfolio.weights.to_numpy() * (1 - self.fee),
            index=portfolio.weights.index.copy(),
            columns=portfolio.weights.columns.copy()
        )
        # TODO: add correct calculus for leveraged residuals
        positions.insert(loc=0, column="CASH_RESIDUALS", value=0)

        portfolio.set_positions(positions)

        return portfolio


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


class CalculateReturns:
    def __init__(self, universe: Universe):
        self.universe = universe

    def __call__(self, portfolio: Portfolio) -> Portfolio:
        prices, positions = align(self.universe.prices, portfolio.positions)

        universe_returns = prices.pct_change().to_numpy()[1:]
        portfolio_returns = (positions[:-1] * universe_returns)

        dead_returns = np.where(
            np.isnan(portfolio_returns) & ~np.isclose(positions[:-1], 0),
            -positions[:-1], 0
        )
        returns = np.nansum(portfolio_returns, axis=1) + np.nansum(dead_returns, axis=1)

        portfolio.set_returns(
            pd.Series(
                returns,
                index=portfolio.positions.index[1:].copy()
            )
        )

        return portfolio
