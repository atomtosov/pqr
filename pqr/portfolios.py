"""
This module contains stuff to create portfolios of stocks. Portfolio is a
rebalancing each period batch of stocks. So, portfolio can be represented as
matrix with zeros on cells, where a stock is not picked, and weights (or lots)
on cells, where a stock is picked into a portfolio.

Portfolio can be constructed relatively or with money. Relative portfolio is a
theoretical portfolio, built up with assumption that all available money is
invested in each period (and each period returns are reinvested). Money
portfolio is a more realistic portfolio, which operates with real balance,
which imposes restrictions on the process of investing: practically never
weights of positions are equal to theoretical weights (because of
indivisibility of stocks). Money portfolio is going close to relative one with
initial balance going to infinity.

Also, portfolio can be winners-minus-losers if you construct a factor model.
This type of portfolio is supported, but without real allocation: positions of
winners and losers portfolios a simply used as positions for the wml-portfolio.
Moreover, this portfolio is always theoretical, as it is assumed as
self-financing portfolio (money from shorts are used to long stocks).

Random portfolios to test performance of your portfolio are also supported, but
only for long-only (not wml) portfolios.
"""

from __future__ import annotations

from typing import Protocol, Tuple, Optional, Any, List, Literal

import numpy as np
import pandas as pd

import pqr.factors

__all__ = [
    'AbstractPortfolio',
    'Portfolio',
    'WmlPortfolio',
    'RandomPortfolio',
    'generate_random_portfolios',
]


class AbstractPortfolio(Protocol):
    """
    Abstract class for portfolios.

    Describes, what fields must be included into concrete portfolios and
    realizes methods for fancy printing.
    """

    name: str
    """Name of the portfolio."""
    picks: pd.DataFrame
    """Matrix with stocks picked into the portfolio."""
    weights: pd.DataFrame
    """Matrix with weights in the portfolio for each stock in each period."""
    positions: pd.DataFrame
    """Matrix with allocated stocks into the portfolio in each period."""
    returns: pd.Series
    """Periodical returns of the portfolio (non-cumulative)."""

    def __repr__(self) -> str:
        return f'{type(self).__name__}({repr(self.name)})'

    def __str__(self) -> str:
        return self.name


class Portfolio(AbstractPortfolio):
    """
    Class for factor portfolios.

    Parameters
    ----------
    name
        Name of the portfolio.
    """

    def __init__(self, name: str = 'portfolio'):
        self.name = name

        self.picks = pd.DataFrame()
        self.weights = pd.DataFrame()
        self.positions = pd.DataFrame()
        self.returns = pd.Series()

    def pick_stocks_by_factor(
            self,
            factor: pqr.factors.Factor,
            thresholds: Tuple[int | float, int | float],
            method: Literal['quantile', 'top', 'time-series'] = 'quantile'
    ) -> Portfolio:
        """
        Picks subset of stocks into the portfolio, choosing them by `factor`.

        Supports 3 methods to pick stocks:

        * quantile
        * top
        * time-series

        Parameters
        ----------
        factor
            Factor to pick stocks into the portfolio.
        thresholds
            Bounds for the set of allowed values of `factor` to pick stocks.
        method
            Method, used to define subset of stocks to be picked on the basis
            of given `thresholds`.
        """

        factor = factor.data

        if method == 'quantile':
            lower_threshold, upper_threshold = np.nanquantile(
                factor, thresholds, axis=1, keepdims=True
            )
        elif method == 'top':
            lower_threshold = np.nanmin(
                factor.apply(pd.Series.nlargest, n=thresholds[1], axis=1),
                axis=1, keepdims=True
            )
            upper_threshold = np.nanmin(
                factor.apply(pd.Series.nlargest, n=thresholds[0], axis=1),
                axis=1, keepdims=True
            )
        else:  # method = 'time-series'
            lower_threshold, upper_threshold = thresholds

        self.picks = (lower_threshold <= factor) & (factor <= upper_threshold)

        return self

    def weigh_by_factor(
            self,
            factor: pqr.factors.Factor
    ) -> Portfolio:
        """
        Weighs the `picks` by `factor`.

        Finds linear weights: simply divides each value in a row by the sum of
        the row. If factor is lower_better (`is_bigger_better` = False), then
        weights are additionally "mirrored".

        Parameters
        ----------
        factor
            Factor to weigh picks.

        Notes
        -----
        Works only for factors with all positive values.
        """

        raw_weights = self.picks * factor.data.loc[self.picks.index[0]:]
        normalizer = np.nansum(raw_weights, axis=1, keepdims=True)

        self.weights = (raw_weights / normalizer).fillna(0)

        return self

    def weigh_equally(self) -> Portfolio:
        """
        Weighs the `picks` equally: all stocks will have the same weights.
        """

        raw_weights = self.picks * np.ones_like(self.picks, dtype=int)
        normalizer = np.nansum(raw_weights, axis=1, keepdims=True)
        self.weights = (raw_weights / normalizer).fillna(0)

        return self

    def scale_weights_by_factor(
            self,
            factor: pqr.factors.Factor,
            target: int | float = 1
    ) -> Portfolio:
        """
        Scale the `weights` by `target` of `factor`.

        Simply divides each value in a row by `target`. If `factor` is
        lower_better (`bigger_better` = False), then leverages is additionally
        "mirrored".

        Parameters
        ----------
        factor
            Factor to scale (leverage) positions weights.
        target
            Target of `factor`.

        Notes
        -----
        Works only for factors with all positive values.
        """

        leveraged_weights = self.weights * factor.data / target

        self.weights = leveraged_weights.fillna(0)

        return self

    def allocate(
            self,
            stock_prices: pd.DataFrame,
            balance: Optional[int | float] = None,
            fee_rate: int | float = 0
    ) -> Portfolio:
        """
        Allocates positions, based on `weights`.

        If `balance` is None:
            positions will be equal to weights with correction on commission.

        If `balance` is number > 0:
            positions will be allocated, using simple greedy algorithm - buy as
            much stocks as possible to be maximum closely to expected weights.

        Parameters
        ----------
        stock_prices
            Prices, representing stock universe.
        balance
            Initial balance of the portfolio.
        fee_rate
            Indicative commission rate for every deal.

        Notes
        -----
        For now allocation with money can lead to negative cash if
        `fee_rate` != 0, but it will be fixed soon.
        """

        if balance is None:
            self._allocate_relatively(stock_prices, fee_rate)
        else:
            self._allocate_with_money(stock_prices, balance, fee_rate)

        self.returns.name = self.name

        return self

    def _allocate_relatively(
            self,
            stock_prices: pd.DataFrame,
            fee_rate: int | float
    ) -> None:
        """
        Allocates positions relatively (theoretically).
        """

        stock_prices = stock_prices.loc[self.weights.index[0]:]
        self.positions = self.weights
        universe_returns = stock_prices.pct_change().shift(-1)
        portfolio_returns = (
            (self.weights * universe_returns).shift().sum(axis=1))

        self.returns = portfolio_returns * (1 - fee_rate)

    def _allocate_with_money(
            self,
            stock_prices: pd.DataFrame,
            balance: int | float,
            fee_rate: int | float
    ) -> None:
        """
        Allocates positions with money.
        """

        stock_prices = stock_prices.loc[self.weights.index[0]:]

        portfolio = self.weights.copy()
        returns = pd.Series(index=portfolio.index)

        portfolio.values[0] = (
                portfolio.values[0] * balance // stock_prices.values[0])
        returns.values[0] = 0
        cash = balance - np.nansum(portfolio.values[0] *
                                   stock_prices.values[0] * (1 + fee_rate))
        prev_balance = balance
        for i in range(1, len(portfolio)):
            w = portfolio.values[i]
            prices = stock_prices.values[i]
            prev_alloc = portfolio.values[i - 1]

            cur_balance = cash + np.nansum(prev_alloc * prices)
            alloc = w * cur_balance // prices

            alloc_diff = alloc - prev_alloc
            cash_diff = -(alloc_diff * prices)
            commission = np.abs(cash_diff) * fee_rate

            cash += np.nansum(cash_diff - commission)

            portfolio.values[i] = alloc
            returns.values[i] = cur_balance / prev_balance - 1

            prev_balance = cur_balance

        self.positions = portfolio
        self.returns = returns


class WmlPortfolio(AbstractPortfolio):
    """
    Class for WML (winners-minus-losers) portfolios.

    This type of portfolio is always treated as theoretical (self-financing),
    and that is why it assumes that given portfolios are relevant: they are
    built with the same balance, or both of them are relative. Actually, it
    simply subtracts `losers_portfolio` from `winners_portfolio`.

    Parameters
    ----------
    winners_portfolio
        Portfolio of winners by some factor. Positions of this portfolio will
        be long-positions for a wml-portfolio.
    losers_portfolio
        Portfolio of losers by the same factor. Positions of this portfolio
        will be short-positions for a wml-portfolio.
    name
        Name of the wml-portfolio.
    """

    def __init__(
            self,
            winners_portfolio: Portfolio,
            losers_portfolio: Portfolio,
            name: str = 'wml'
    ):
        self.name = name

        self.picks = (winners_portfolio.picks.astype(int) -
                      losers_portfolio.picks.astype(int))
        self.weights = winners_portfolio.weights - losers_portfolio.weights
        self.positions = (winners_portfolio.positions -
                          losers_portfolio.positions)
        self.returns = winners_portfolio.returns - losers_portfolio.returns
        self.returns.name = self.name


class RandomPortfolio(AbstractPortfolio):
    """
    Class for random portfolios, replicating picks of a portfolio.

    It implements the delegation pattern, so it has the same interface as
    Portfolio with the only one exception: the method
    ``pick_stocks_by_factor()`` is replaced with the method
    ``pick_stocks_randomly()``.

    Parameters
    ----------
    name
        Name of the random portfolio.
    random_seed
        Random seed to make random deterministic.
    """

    def __init__(
            self,
            name: str = 'random',
            random_seed: Optional[int] = None
    ):
        self._portfolio = Portfolio(name)

        if random_seed is not None:
            np.random.seed(random_seed)

    def __getattr__(self, name: str) -> Any:
        if name == 'pick_stocks_by_factor':
            raise AttributeError('\'RandomPortfolio\' object has no attribute '
                                 '\'pick_stocks_by_factor\'')
        return getattr(self._portfolio, name)

    def pick_stocks_randomly(
            self,
            picks: pd.DataFrame,
            mask: Optional[pd.DataFrame] = None
    ) -> RandomPortfolio:
        """
        Pick stocks randomly, but in the same quantity as in the `picks`.

        In each period collects number of picked stocks and randomly pick the
        same amount of stocks into a random portfolio.

        Parameters
        ----------
        picks
            Picks to replicate by the random portfolio.
        mask
            Matrix to prohibit pick stock during periods, when they are not
            really available for trading. Also can be used to filter stock
            universe.
        """

        picks = picks.copy().astype(float)
        if mask is not None:
            picks[~mask] = np.nan

        def random_pick(row: np.ndarray,
                        indices=np.indices((picks.shape[1],))[0]):
            picked = (row > 0).sum()
            choice = np.random.choice(indices[~np.isnan(row)], picked)
            random_picks = np.zeros_like(row, dtype=bool)
            random_picks[choice] = True
            return random_picks

        self._portfolio.picks = pd.DataFrame(
            np.apply_along_axis(random_pick, axis=1, arr=picks.values),
            index=picks.index, columns=picks.columns
        )

        return self

    def weigh_by_factor(
            self,
            factor: pqr.factors.Factor
    ) -> RandomPortfolio:
        """
        See Portfolio.weigh_by_factor().
        """

        self._portfolio.weigh_by_factor(factor)

        return self

    def weigh_equally(self) -> RandomPortfolio:
        """
        See Portfolio.weigh_equally().
        """

        self._portfolio.weigh_equally()

        return self

    def scale_weights_by_factor(
            self,
            factor: pqr.factors.Factor,
            target: int | float = 1
    ) -> RandomPortfolio:
        """
        See Portfolio.scale_weights_by_factor().
        """

        self._portfolio.scale_weights_by_factor(factor, target)

        return self

    def allocate(
            self,
            stock_prices: pd.DataFrame,
            balance: Optional[int | float] = None,
            fee_rate: int | float = 0
    ) -> RandomPortfolio:
        """
        See Portfolio.allocate().
        """

        self._portfolio.allocate(stock_prices, balance, fee_rate)

        return self


def generate_random_portfolios(
        stock_prices: pd.DataFrame,
        portfolio: Portfolio,
        mask: Optional[pd.DataFrame] = None,
        weighting_factor: Optional[pqr.factors.Factor] = None,
        scaling_factor: Optional[pqr.factors.Factor] = None,
        target: int | float = 1,
        balance: Optional[int | float] = None,
        fee_rate: int | float = 0,
        n: int = 100,
        random_seed: Optional[int] = None
) -> List[RandomPortfolio]:
    """
    Creates `n` random portfolios, replicating portfolio `picks`.

    Parameters
    ----------
    stock_prices
        Prices, representing stock universe.
    portfolio
        Portfolio to be replicated by random ones.
    mask
        Matrix to filter stock universe. By default, stocks which are not
        available for trading are excluded, so there is no necessity to pass
        this mask, if specific filter is not used.
    weighting_factor
        Factor, weighting positions. If not given, just weigh equally.
    scaling_factor
        Factor to scale (leverage) positions weights.
    target
        Target of `scaling_factor`.
    balance
        Initial balance of the portfolio.
    fee_rate
        Indicative commission rate for every deal.
    n
        How many random portfolios to be created.
    random_seed
        Random seed to make random deterministic.
    """

    np.random.seed(random_seed)

    picks = portfolio.picks.astype(float)
    picks[stock_prices.isna()] = np.nan
    if mask is not None:
        picks[~mask] = np.nan

    portfolios = []
    for i in range(n):
        random_portfolio = RandomPortfolio()
        random_portfolio.pick_stocks_randomly(picks)
        if weighting_factor is not None:
            random_portfolio.weigh_by_factor(weighting_factor)
        else:
            random_portfolio.weigh_equally()
        if scaling_factor is not None:
            random_portfolio.scale_weights_by_factor(scaling_factor, target)
        random_portfolio.allocate(stock_prices, balance, fee_rate)

        portfolios.append(random_portfolio)

    return portfolios


def _ranked_reverse(data: pd.DataFrame) -> pd.DataFrame:
    """
    Algorithm, mirroring weights for lower_better factors.
    """

    data = data.copy()
    straight_sort = np.argsort(data.values, axis=1)
    reversed_sort = np.fliplr(straight_sort)
    for i in range(len(data)):
        data.values[i, straight_sort[i]] = data.values[i, reversed_sort[i]]

    return data
