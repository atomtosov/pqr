import abc
from typing import Union, Optional

import numpy as np
import pandas as pd

import pqr.factors
import pqr.thresholds

__all__ = [
    'AbstractPortfolio',
    'Portfolio',
    'WmlPortfolio',
]


class AbstractPortfolio(abc.ABC):
    """
    Abstract base class for portfolios of assets.
    """

    positions: pd.DataFrame
    returns: pd.Series
    trading_start: pd.Timestamp

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

               picking_factor: Optional[pqr.factors.Factor] = None,
               picking_thresholds:
               pqr.thresholds.Thresholds = pqr.thresholds.Thresholds(),
               looking_period: int = 1,
               lag_period: int = 0,
               holding_period: int = 1,

               filtering_factor: Optional[pqr.factors.Factor] = None,
               filtering_thresholds:
               pqr.thresholds.Thresholds = pqr.thresholds.Thresholds(),
               filtering_looking_period: int = 1,
               filtering_lag_period: int = 0,
               filtering_holding_period: int = 1,

               weighting_factor: Optional[pqr.factors.Factor] = None,
               weighting_looking_period: int = 1,
               weighting_lag_period: int = 0,
               weighting_holding_period: int = 1,

               scaling_factor: Optional[pqr.factors.Factor] = None,
               scaling_target: Union[int, float] = 1,
               scaling_looking_period: int = 1,
               scaling_lag_period: int = 0,
               scaling_holding_period: int = 1) -> 'Portfolio':
        """
        Allocate money in stocks, making decisions on the basis of factors.

        Parameters
        ----------
        prices : pd.DataFrame
            Dataframe, representing stock universe by prices.
        picking_factor : pqr.factors.Factor
            Factor, used to pick stocks from (filtered) stock universe.
        picking_thresholds : pqr.thresholds.Thresholds
            Thresholds, limiting picks of picking factor.
        looking_period : int, default=1
            Looking back on picking_factor period.
        lag_period : int, default=0
            Delaying period to react on picking_factor picks.
        holding_period : int, default=1
            Number of periods to hold each pick of picking_factor.
        filtering_factor : pqr.factors.Factor, optional
            Factor, filtering stock universe. If not given, just not filter at
            all.
        filtering_thresholds : pqr.thresholds.Thresholds, default=Thresholds()
            Thresholds, limiting filtering of filtering_factor.
        filtering_looking_period : int, default=1
            Looking back on filtering_factor period.
        filtering_lag_period : int, default=0
            Delaying period to react on filtering_factor filters.
        filtering_holding_period : int, default=1
            Number of periods to hold each filter of filtering_factor.
        weighting_factor : pqr.factors.Factor, optional
            Factor, weighting picks of picking_factor. If not given, just weigh
            equally.
        weighting_looking_period : int, default=1
            Looking back on weighting_factor period.
        weighting_lag_period : int, default=0
            Delaying period to react on weighting_factor weights.
        weighting_holding_period : int, default=1
            Number of periods to hold each weight of weighting_factor.
        scaling_factor : pqr.factors.Factor, optional
            Factor, scaling (leveraging) weights. If not given just not scale 
            at all.
        scaling_target : int, float, default=1
            Target to scale weights by scaling factor.
        scaling_looking_period : int, default=1
            Looking back on scaling_factor period.
        scaling_lag_period : int, default=0
            Delaying period to react on scaling_factor leverages.
        scaling_holding_period : int, default=1
            Number of periods to hold each leverage of scaling_factor.
            
        Returns
        -------
        Portfolio
            The same object, but with filled positions and returns.
        """

        i_trading_start = looking_period + lag_period + picking_factor.dynamic
        self.trading_start = prices.index[i_trading_start]
        self.thresholds = picking_thresholds

        filtered_prices = pqr.factors.filter(prices, filtering_factor,
                                             filtering_thresholds,
                                             filtering_looking_period,
                                             filtering_lag_period,
                                             filtering_holding_period)
        picks = pqr.factors.pick(filtered_prices, picking_factor,
                                 picking_thresholds, looking_period,
                                 lag_period, holding_period)
        weights = pqr.factors.weigh(picks, weighting_factor,
                                    weighting_looking_period,
                                    weighting_lag_period,
                                    weighting_holding_period)
        weights = pqr.factors.scale(weights, scaling_factor, scaling_target,
                                    scaling_looking_period, scaling_lag_period,
                                    scaling_holding_period)

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
