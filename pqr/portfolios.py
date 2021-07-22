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


import dataclasses
from typing import Optional, Union, List

import numpy as np
import pandas as pd

import pqr.factors
import pqr.thresholds


__all__ = [
    'Portfolio',
    'factor_portfolio',
    'wml_portfolio',
    'random_portfolios',
]


@dataclasses.dataclass(frozen=True, repr=False)
class Portfolio:
    """
    Class for various types of portfolios.

    Parameters
    ----------
    positions
        Positions of a portfolio in each period of trading.
    returns
        Period-to-period returns of a portfolio.
    balance
        Initial cash available to build a portfolio. If not passed, the
        portfolio is constructed relatively.
    fee_rate
        Indicative commission of each trade.
    name
        Name of a portfolio, it is used for fancy printing.
    """

    positions: pd.DataFrame
    """Positions of the portfolio."""
    returns: pd.Series
    """Period-to-period returns of the portfolio."""
    balance: Optional[Union[int, float]] = None
    """Initial cash available for investing."""
    fee_rate: Union[int, float] = 0
    """Indicative commission rate."""
    name: str = 'portfolio'
    """Name of the portfolio."""

    def __str__(self) -> str:
        return self.name

    @property
    def cumulative_returns(self) -> pd.Series:
        """Cumulative returns of the portfolio."""

        return (1 + self.returns).cumprod() - 1


def factor_portfolio(
        stock_prices: pd.DataFrame,
        factor: pd.DataFrame,
        thresholds: pqr.thresholds.Thresholds,
        filtering_factor: Optional[pd.DataFrame] = None,
        filtering_thresholds:
        pqr.thresholds.Thresholds = pqr.thresholds.Thresholds(),
        weighting_factor: Optional[pd.DataFrame] = None,
        weighting_factor_is_bigger_better: bool = True,
        scaling_factor: Optional[pd.DataFrame] = None,
        scaling_factor_is_bigger_better: bool = True,
        scaling_target: Union[int, float] = 1,
        balance: Union[int, float] = None,
        fee_rate: Union[int, float] = 0,
        name: str = 'portfolio'
) -> Portfolio:
    """
    Constructs a portfolio by factors.

    At first, filtrates stock universe and passes it to the `factor`, which 
    task is to pick stocks into the portfolio. Then, these picks are weighted
    and (optionally) scaled. If balance is passed, then also simulates real
    allocation of money by scaled weights.

    Parameters
    ----------
    stock_prices
        Prices, representing stock universe.
    factor
        Factor, used to pick stocks from (filtered) stock universe.
    thresholds
        Thresholds, limiting `factor`.
    filtering_factor
        Factor, filtering stock universe. If not given, just not filter at
        all.
    filtering_thresholds
        Thresholds, limiting `filtering_factor`.
    weighting_factor
        Factor, weighting picks of `factor`. If not given, just weigh
        equally.
    weighting_factor_is_bigger_better
        Whether bigger values of `weighting_factor` will lead to bigger weights 
        for a position or on the contrary to lower.
    scaling_factor
        Factor, scaling (leveraging) weights. If not given just not scale 
        at all.
    scaling_factor_is_bigger_better
        Whether bigger values of `scaling_factor` will lead to bigger leverage 
        for a position or on the contrary to lower.
    scaling_target
        Target to scale weights by scaling factor.
    balance
        Initial balance of portfolio.
    fee_rate
        Commission rate for every deal.
    name
        Name of the portfolio, it is used for fancy printing.
    """

    stock_universe = pqr.factors.filtrate(stock_prices, filtering_factor,
                                          filtering_thresholds)
    picks = pqr.factors.select(stock_universe, factor, thresholds)
    weights = pqr.factors.weigh(picks, weighting_factor,
                                weighting_factor_is_bigger_better)
    weights = pqr.factors.scale(weights, scaling_factor, scaling_target,
                                scaling_factor_is_bigger_better)

    stock_prices = stock_prices.loc[weights.index[0]:]

    if balance is None:
        return _relative_portfolio(stock_prices, weights, fee_rate, name)
    else:
        return _money_portfolio(stock_prices, weights, balance, fee_rate, name)


def wml_portfolio(winners: Portfolio,
                  losers: Portfolio) -> Portfolio:
    """
    Constructs the WML-portfolio (winners-minus-losers).

    This type of portfolio is always treated as theoretical, so, it cannot be
    built up with balance. In the case if you pass 2 portfolios, built up with
    money (`balance` != None), the constructed wml-portfolio assumed as
    self-financing.

    Parameters
    ----------
    winners
        Portfolio of winners by some factor. Positions of this portfolio will
        be long-positions for a wml-portfolio.
    losers
        Portfolio of losers by the same factor. Positions of this portfolio
        will be short-positions for a wml-portfolio.
    """

    positions = winners.positions - losers.positions
    returns = winners.returns - losers.returns
    return Portfolio(positions, returns, None, winners.fee_rate,
                     'wml_portfolio')


def random_portfolios(
        stock_prices: pd.DataFrame,
        portfolio: Portfolio,
        filtering_factor: Optional[pd.DataFrame] = None,
        filtering_thresholds:
        pqr.thresholds.Thresholds = pqr.thresholds.Thresholds(),
        weighting_factor: Optional[pd.DataFrame] = None,
        weighting_factor_is_bigger_better: bool = True,
        scaling_factor: Optional[pd.DataFrame] = None,
        scaling_factor_is_bigger_better: bool = True,
        scaling_target: Union[int, float] = 1,
        n: int = 100,
        random_seed: Optional[int] = None,
) -> List[Portfolio]:
    """
    Creates `n` random portfolios, replicating `portfolio` positions.
    
    In each period collects number of picked stocks and randomly pick the same
    number of stocks into a random portfolio.
    
    Parameters
    ----------
    stock_prices
        Prices, representing stock universe.
    portfolio
        Portfolio to be replicated by random ones.
    filtering_factor
        Factor, filtering stock universe. If not given, just not filter at
        all.
    filtering_thresholds
        Thresholds, limiting `filtering_factor`.
    weighting_factor
        Factor, weighting picks of `factor`. If not given, just weigh
        equally.
    weighting_factor_is_bigger_better
        Whether bigger values of `weighting_factor` will lead to bigger weights 
        for a position or on the contrary to lower.
    scaling_factor
        Factor, scaling (leveraging) weights. If not given just not scale 
        at all.
    scaling_factor_is_bigger_better
        Whether bigger values of `scaling_factor` will lead to bigger leverage 
        for a position or on the contrary to lower.
    scaling_target
        Target to scale weights by scaling factor.
    n
        How many random portfolios to be created.
    random_seed
        Random seed to make random deterministic.
        
    Notes
    -----
    Works only for long-only portfolios.
    """

    positions = portfolio.positions.copy()
    stock_prices = stock_prices[positions.index[0]:]

    stock_universe = pqr.factors.filtrate(stock_prices, filtering_factor,
                                          filtering_thresholds)
    positions.values[stock_universe.isna()] = np.nan

    def random_pick(row: np.ndarray,
                    rng=np.random.default_rng(random_seed),
                    indices=np.indices((positions.shape[1],))[0]):
        picked = (row > 0).sum()
        choice = rng.choice(indices[~np.isnan(row)], picked)
        picks = np.zeros_like(row, dtype=bool)
        picks[choice] = True
        return picks

    portfolios = []
    for i in range(n):
        picks = pd.DataFrame(
            np.apply_along_axis(random_pick, 1, positions.values),
            index=positions.index, columns=positions.columns)
        weights = pqr.factors.weigh(picks, weighting_factor,
                                    weighting_factor_is_bigger_better)
        weights = pqr.factors.scale(weights, scaling_factor, scaling_target,
                                    scaling_factor_is_bigger_better)
        if portfolio.balance is None:
            portfolios.append(
                _relative_portfolio(stock_prices, weights, portfolio.fee_rate,
                                    f'random ~ {portfolio.name}'))
        else:
            portfolios.append(
                _money_portfolio(stock_prices, weights, portfolio.balance,
                                 portfolio.fee_rate,
                                 f'random ~ {portfolio.name}'))

    return portfolios


def _relative_portfolio(
        stock_prices: pd.DataFrame,
        weights: pd.DataFrame,
        fee_rate: Union[int, float],
        name: str
) -> Portfolio:
    """
    Constructs the portfolio relatively (without real money rebalancing).
    """

    positions = weights
    returns = (positions * stock_prices.pct_change().shift(-1)
               ).shift().sum(axis=1)
    # TODO: add commission
    return Portfolio(positions, returns, None, fee_rate, name)


def _money_portfolio(
        stock_prices: pd.DataFrame,
        weights: pd.DataFrame,
        balance: Union[int, float],
        fee_rate: Union[int, float],
        name: str
) -> Portfolio:
    """
    Constructs the portfolio with real money rebalancing.
    """

    raise NotImplementedError('money portfolio does not work for now')
