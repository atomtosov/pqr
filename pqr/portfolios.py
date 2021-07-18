import abc
from typing import Union, Optional, Callable

import numpy as np
import pandas as pd

import pqr.factors
import pqr.thresholds

__all__ = [
    'AbstractPortfolio',
    'Portfolio',
    'WmlPortfolio',
    'RandomPortfolio',
]


class AbstractPortfolio(abc.ABC):
    """
    Abstract base class for portfolios of assets.
    """

    positions: pd.DataFrame
    """Positions of portfolio in each period."""
    returns: pd.Series
    """Period-to-period returns of portfolio."""
    trading_start: pd.Timestamp
    """Date of starting trading."""

    def __repr__(self) -> str:
        return f'{type(self).__name__}()'

    @abc.abstractmethod
    def invest(self, *args, **kwargs) -> 'AbstractPortfolio':
        """
        Filling the portfolio with assets. In this method positions and
        returns must be calculated, and also trading_start should be set. Must
        return the same object (self).
        """

    @property
    def cumulative_returns(self) -> pd.Series:
        """
        pd.Series : Cumulative returns of portfolio.
        """

        return (1 + self.returns).cumprod() - 1


class Portfolio(AbstractPortfolio):
    """
    Class for portfolio, allocating money in stocks by factors.

    Parameters
    ----------
    balance : int, float, optional
        Initial balance of portfolio. If not given, used relative portfolio.
    fee_rate : int, float, default=0
        Fee rate for every deal.
    """

    thresholds: pqr.thresholds.Thresholds

    def __init__(self,
                 balance: Optional[Union[int, float]] = None,
                 fee_rate: Union[int, float] = 0):
        self.balance = balance
        self.fee_rate = fee_rate

    def __repr__(self) -> str:
        thresholds = getattr(self, 'thresholds', '')
        return f'{type(self).__name__}({thresholds})'

    def invest(self,
               prices: pd.DataFrame,
               filter: Optional[pqr.factors.Filter] = pqr.factors.Filter(),
               picker: Optional[pqr.factors.Picker] = pqr.factors.Picker(),
               weigher: Optional[pqr.factors.Weigher] = pqr.factors.Weigher(),
               scaler: Optional[
                   pqr.factors.Scaler] = pqr.factors.Scaler()) -> 'Portfolio':
        """
        Allocates balance into stocks from stock universe.

        At first, stock universe is filtered by `filter`, then `picker` is used
        to pick stocks from filtered stock universe. After the picks are used
        as input for `weigher`, and weights are scaled by `scaler`.

        Parameters
        ----------
        prices : pd.DataFrame
            Prices, representing stock universe.
        filter : pqr.factors.Filter, default=Filter(None)
            Callable object, which can filter stock universe. By default used
            no-filter.
        picker : pqr.factors.Picker, default=Picker(None)
            Callable object, which can pick stocks from stock universe. By
            default used all-pick.
        weigher : pqr.factors.Weigher, default=Weigher(None)
            Callable object, which can weigh positions of portfolio (picks). By
            default used equal-weights.
        scaler : pqr.factors.Picker, default=Scaler(None)
            Callable object, which can scale weights. By default used
            no-scaler.

        Returns
        -------
        Portfolio
            The same object, but with filled portfolios. Transformation is done
            inplace.
        """

        i_trading_start = (picker.looking_period + picker.lag_period +
                           picker.factor.dynamic)
        self.trading_start = prices.index[i_trading_start]
        self.thresholds = picker.thresholds

        filtered_prices = filter(prices)
        picks = picker(filtered_prices)
        weights = weigher(picks)
        weights = scaler(weights)

        if self.balance is None:
            self._allocate_relative(prices, weights)
        else:
            self._allocate_money(prices, weights)

        return self

    def _allocate_relative(self,
                           prices: pd.DataFrame,
                           weights: pd.DataFrame) -> None:
        self.positions = weights
        self.returns = (self.positions * prices.pct_change().shift(-1)
                        ).shift().sum(axis=1)

    def _allocate_money(self,
                        prices: pd.DataFrame,
                        weights: pd.DataFrame) -> None:
        self.positions = pd.DataFrame(np.zeros_like(weights.values, dtype=int),
                                      index=weights.index,
                                      columns=weights.columns)
        cash = self.balance
        equity_curve = pd.Series([self.balance] * weights.shape[1],
                                 index=weights.index)
        for i in range(1, len(weights)):
            prev_prices = prices.values[i - 1]
            prev_allocation = self.positions.values[i - 1]
            current_prices = prices.values[i]
            current_weights = weights.values[i]
            current_balance = equity_curve.values[i - 1]
            current_allocation = (
                    current_weights * current_balance // prev_prices)
            allocation_diff = current_allocation - prev_allocation
            rebalancing_cash = allocation_diff * prev_prices
            rebalancing_commission = (
                    np.nansum(np.abs(rebalancing_cash)) * self.fee_rate)
            cash -= np.nansum(rebalancing_cash) - rebalancing_commission
            self.positions.values[i] = current_allocation
            equity_curve.values[i] = np.nansum(
                current_allocation * current_prices) + cash

        self.returns = equity_curve.pct_change()


class WmlPortfolio(AbstractPortfolio):
    """
    Class for theoretical WML-portfolios (winners-minus-losers). Winners are
    bought in long, whereas losers are shorted. So, represents risk-neutral
    arbitration portfolio.
    """

    def invest(self,
               winners: AbstractPortfolio,
               losers: AbstractPortfolio) -> 'WmlPortfolio':
        """

        Parameters
        ----------
        winners : Portfolio
            Portfolio of winners by some factor. These assets will be longed.
        losers : Portfolio
            Portfolio of losers by the same factor. These assets will be
            shorted.

        Returns
        -------
        WmlPortfolio
            The same object, but with filled positions and returns.
        """

        self.positions = winners.positions - losers.positions
        self.returns = winners.returns - losers.returns
        self.trading_start = min([winners.trading_start, losers.trading_start])

        return self


class RandomPortfolio(AbstractPortfolio):
    target_value: Union[int, float]

    def __repr__(self) -> str:
        target_value = getattr(self, 'target_value', '')
        return f'{type(self).__name__}({target_value:.2f})'

    def invest(self,
               prices: pd.DataFrame,
               portfolio: Portfolio,
               target: Callable[[AbstractPortfolio], Union[int, float]],
               filter: Optional[pqr.factors.Filter] = pqr.factors.Filter(),
               weigher: Optional[pqr.factors.Weigher] = pqr.factors.Weigher(),
               scaler: Optional[pqr.factors.Scaler] = pqr.factors.Scaler()
               ) -> 'RandomPortfolio':
        random_data = pd.DataFrame(np.random.random(portfolio.positions.shape),
                                   index=portfolio.positions.index,
                                   columns=portfolio.positions.columns)
        random_factor = pqr.factors.Factor(random_data)
        random_picker = pqr.factors.Picker(
            random_factor, portfolio.thresholds)
        random_portfolio = Portfolio(portfolio.balance,
                                     portfolio.fee_rate)
        random_portfolio.invest(prices, filter, random_picker, weigher, scaler)
        random_portfolio.positions[:portfolio.trading_start] = 0
        random_portfolio.returns[:portfolio.trading_start] = 0

        self.positions = random_portfolio.positions
        self.returns = random_portfolio.returns
        self.trading_start = portfolio.trading_start

        self.target_value = target(self)

        return self
