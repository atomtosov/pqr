"""
This module contains stuff to create portfolios of stocks. Portfolio is a rebalancing each period
batch of stocks. So, portfolio can be represented as matrix with zeros on cells, where a stock is
not picked, and weights (or lots) on cells, where a stock is picked into a portfolio.

Portfolio can be constructed relatively or with money. Relative portfolio is a theoretical
portfolio, built up with assumption that all available money is invested in each period (and each
period returns are reinvested). Money portfolio is a more realistic portfolio, which operates with
real balance, which imposes restrictions on the process of investing: practically never weights of
positions are equal to theoretical weights (because of indivisibility of stocks). Money portfolio is
going close to relative one with initial balance going to infinity.
"""

import pandas as pd
import numpy as np

from .utils import align


__all__ = [
    'Portfolio',
    'generate_random_portfolios',
]


class Portfolio:
    """Class for factor portfolios.

    Parameters
    ----------
    name : str, default='portfolio'
        Name of the portfolio.
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

    def __init__(self, name='portfolio'):
        self.name = name

        self.picks = pd.DataFrame()
        self.weights = pd.DataFrame()
        self.positions = pd.DataFrame()
        self.returns = pd.Series(dtype=float)

    def __repr__(self):
        return f'Portfolio({repr(self.name)})'

    def __str__(self):
        return self.name

    def pick_all(self, stock_prices, mask=None, direction='long'):
        """Picks all available for trading stocks for `direction` positions.

        Parameters
        ----------
        stock_prices : pd.DataFrame
            Prices, representing stock universe.
        mask : pd.DataFrame, optional
            Mask to filter stock universe.
        direction: {'long', 'short'}
            Whether to buy or to sell short picked stocks.

        Returns
        -------
        Portfolio
            Portfolio with filled picks.
        """

        picks = ~stock_prices.isna()
        if mask is not None:
            picks, mask = align(picks, mask)
            picks &= mask
        self.picks = picks.astype(int)

        if direction == 'short':
            self.picks *= -1

        return self

    def pick_by_factor(self, factor, thresholds, better='more', method='quantile'):
        """Picks subset of stocks into the portfolio, choosing them by `factor`.

        Supports 3 methods to pick stocks:

        * quantile
        * top
        * time-series

        Parameters
        ----------
        factor : Factor
            Factor to pick stocks into the portfolio.
        thresholds : tuple of int or float
            Bounds for the set of allowed values of `factor` to pick stocks.
        better: {'more', 'less'}, default='more'
            Whether bigger values of factor are treated as better to pick or in contrary as better to 
            avoid. 
        method : {'quantile', 'top', 'time-series'}, default='quantile'
            Method, used to define subset of stocks to be picked on the basis of given `thresholds`.
        
        Returns
        -------
        Portfolio
            Portfolio with filled picks.
        """

        if method == 'quantile':
            if better == 'more':
                thresholds = (1 - thresholds[1], 1 - thresholds[0])

            lower_threshold, upper_threshold = np.nanquantile(
                factor.data, thresholds, axis=1, keepdims=True
            )
        elif method == 'top':
            if better == 'more':
                top_func = pd.Series.nlargest
                bound_func = np.nanmin
            else:  # better == 'less'
                top_func = pd.Series.nsmallest
                bound_func = np.nanmax

            lower_threshold = bound_func(
                factor.data.apply(top_func, n=thresholds[1], axis=1),
                axis=1, keepdims=True
            )
            upper_threshold = bound_func(
                factor.data.apply(top_func, n=thresholds[0], axis=1),
                axis=1, keepdims=True
            )
        else:  # method = 'time-series'
            lower_threshold, upper_threshold = thresholds

        self.picks = ((lower_threshold <= factor.data) & (factor.data <= upper_threshold)).astype(int)

        return self

    def pick_wml(self, winners, losers):
        """Constructs long-short picks from 2 long-portfolios.

        Parameters
        ----------
        winners : Portfolio
            A portfolio, which picks will be long-positions.
        losers : Portfolio
            A portfolio, which picks will be short-positions.

        Returns
        -------
        Portfolio
            Portfolio with filled picks.
        """

        self.picks = (winners.picks - losers.picks).fillna(0)

        return self

    def pick_randomly(self, picks, mask=None):
        """Pick stocks randomly, but in the same quantity as in the `picks`.

        In each period collects number of picked stocks and randomly pick the same amount of stocks
        into a random portfolio.

        Parameters
        ----------
        picks : pd.DataFrame
            Picks to replicate by the random portfolio.
        mask : pd.DataFrame, optional
            Matrix to prohibit pick stock during periods, when they are not really available for trading. 
            Also can be used to filter stock universe.

        Returns
        -------
        Portfolio
            Portfolio with filled picks.
        """

        picks = picks.copy().astype(float)
        if mask is not None:
            picks[~mask] = np.nan

        def random_pick(row, indices=np.indices((picks.shape[1],))[0]):
            random_picks = np.zeros_like(row, dtype=int)

            long = (row == 1).sum()
            long_choice = np.random.choice(indices[~np.isnan(row)], long)
            random_picks[long_choice] = 1

            short = (row == -1).sum()
            short_choice = np.random.choice(indices[~np.isnan(row) & (random_picks == 0)], short)
            random_picks[short_choice] = -1
            return random_picks

        self.picks = pd.DataFrame(
            np.apply_along_axis(random_pick, axis=1, arr=picks.values),
            index=picks.index, columns=picks.columns
        ).astype(int)

        return self

    def pick_ideally(self, stock_prices, portfolio, mask=None):
        """Picks stocks ideally in the same quantity as in the `portfolio`.

        In each period simply look forward to stock returns and long stocks with the best performance and
        short stocks with the worst into an "ideal" portfolio.

        Parameters
        ----------
        stock_prices : pd.DataFrame
            Prices, representing stock universe.
        portfolio : Portfolio
            Portfolio to be replicated by "ideal" portfolio.
        mask : pd.DataFrame, optional
            Matrix of True/False, where True means that a stock can be picked in that period and
            False - that a stock cannot be picked.

        Returns
        -------
        Portfolio
            Portfolio with filled picks.
        """

        stock_returns = stock_prices.pct_change().shift(-1)
        if mask is not None:
            stock_returns[~mask] = np.nan
        stock_returns, picks = align(stock_returns, portfolio.picks)

        ideal_picks = pd.DataFrame(np.zeros_like(picks), index=picks.index, columns=picks.columns)
        for date in picks.index:
            row = picks.loc[date]
            returns = stock_returns.loc[date]

            long = (row == 1).sum()
            short = (row == -1).sum()
            long_picks = returns.nlargest(long).index
            short_picks = returns.nsmallest(short).index

            ideal_picks.loc[date, long_picks] = 1
            ideal_picks.loc[date, short_picks] = -1
        
        self.picks = ideal_picks

        return self

    def filter(self, mask):
        """Filters `picks` by given `mask`.

        Simply deletes (replaces with 0) cells, where the `mask` equals to False.

        Parameters
        ----------
        mask : pd.DataFrame
            Matrix of True/False, where True means that a pick should remain in `portfolio` and
            False - that a value should be deleted.

        Returns
        -------
        Portfolio
            Portfolio with transformed picks.
        """

        self.picks, mask = align(self.picks, mask)

        self.picks[~mask] = 0

        return self

    def weigh_equally(self):
        """Weighs the `picks` equally: all stocks will have the same weights.

        Returns
        -------
        Portfolio
            Portfolio with filled weights.
        """

        weights = self.picks * np.ones_like(self.picks, dtype=float)
        long, short = weights == 1, weights == -1
        weights[long] /= np.nansum(weights[long], axis=1, keepdims=True)
        weights[short] /= -np.nansum(weights[short], axis=1, keepdims=True)

        self.weights = weights

        return self

    def weigh_by_factor(self, factor):
        """Weighs the `picks` by `factor`.

        Finds linear weights: simply divides each value in a row by the sum of the row.

        Parameters
        ----------
        factor : Factor
            Factor to weigh picks.

        Returns
        -------
        Portfolio
            Portfolio with filled weights.
        """

        picks, factor_data = align(self.picks, factor.data)

        weights = picks * factor_data
        long, short = weights > 0, weights < 0
        weights[long] /= np.nansum(weights[long], axis=1, keepdims=True)
        weights[short] /= -np.nansum(weights[short], axis=1, keepdims=True)

        self.weights = weights.fillna(0)

        return self

    def scale_by_factor(self, factor, target=1, leverage_limits=(-np.inf, np.inf), better='more'):
        """Scale `weights` by `target` of `factor`.

        Simply divides each value in a row by `target` if `better` equals to 'more', otherwise `target` is
        simply divided by a row of `factor` values.

        Parameters
        ----------
        factor : Factor
            Factor to scale (leverage) positions weights.
        target : array_like, default=1
            Target of `factor`.
        leverage_limits : tuple of int or float, default=(-np.inf, np.inf)
            Thresholds to limit min and max leverage for positions weights.
        better : {'more', 'less'}, default='less'
            Whether exceeding the `target` will result in a higher leverage or a lower.

        Returns
        -------
        Portfolio
            Portfolio with scaled weights.

        Notes
        -----
        Works only for factors with all positive values.
        """

        weights, factor_data = align(self.weights, factor.data)

        if better == 'more':
            leverage = factor_data / target
        else:
            leverage = target / factor_data
        weights = weights.multiply(leverage, axis=0)

        total_leverage = weights.sum(axis=1)
        lower_limit, upper_limit = leverage_limits
        exceed_lower = total_leverage < lower_limit
        exceed_upper = total_leverage > upper_limit
        weights[exceed_lower] = weights[exceed_lower].divide(total_leverage[exceed_lower] / lower_limit, axis=0)
        weights[exceed_upper] = weights[exceed_upper].divide(total_leverage[exceed_upper] / upper_limit, axis=0)

        self.weights = weights.fillna(0)

        return self

    def allocate(self, stock_prices, balance=None, fee_rate=0):
        """Allocates positions, based on `weights`.

        If `balance` is None:
            positions will be equal to weights with correction on commission.

        If `balance` is number > 0:
            positions will be allocated, using simple greedy algorithm - buy as
            much stocks as possible to be maximum closely to expected weights.

        Parameters
        ----------
        stock_prices : pd.DataFrame
            Prices, representing stock universe.
        balance : int or float, optional
            Initial balance of the portfolio.
        fee_rate : int or float, default=0
            Indicative commission rate for every deal.

        Returns
        -------
        Portfolio
            Portfolio with filled positions and returns.
        """

        if balance is None:
            self._allocate_relatively(stock_prices, fee_rate)
        else:
            self._allocate_with_money(stock_prices, balance, fee_rate)

        self.returns.name = self.name

        return self

    def _allocate_relatively(self, stock_prices, fee_rate):
        weights, stock_prices = align(self.weights, stock_prices)
        self.positions = weights * (1 - fee_rate)
        universe_returns = stock_prices.pct_change().shift(-1)
        portfolio_returns = (self.positions * universe_returns).shift().sum(axis=1)

        self.returns = portfolio_returns

    def _allocate_with_money(self, stock_prices, balance, fee_rate):
        weights, stock_prices = align(self.weights, stock_prices)

        positions = pd.DataFrame(index=weights.index, columns=weights.columns)
        equity = pd.Series(index=positions.index, dtype=float)
        equity.iloc[0] = balance

        cash = balance
        for i in range(len(weights)):
            w = weights.values[i]
            prices = stock_prices.values[i]
            prev_alloc = np.zeros_like(w) if i == 0 else positions.values[i - 1]

            current_balance = cash + np.nansum(prev_alloc * prices)

            ideal_allocation = np.nan_to_num(w * current_balance / prices).astype(int)
            ideal_allocation_diff = ideal_allocation - prev_alloc
            ideal_cash_diff = -(ideal_allocation_diff * prices)
            ideal_commission = np.nansum(np.abs(ideal_cash_diff) * fee_rate)

            max_allowed_capital = current_balance - ideal_commission

            allocation = np.nan_to_num(w * max_allowed_capital / prices).astype(int)
            allocation_diff = allocation - prev_alloc
            cash_diff = -(allocation_diff * prices)
            commission = np.nansum(np.abs(cash_diff) * fee_rate)

            cash += np.nansum(cash_diff) - commission

            positions.iloc[i] = allocation
            equity.iloc[i] = current_balance - commission

        self.positions = positions
        self.returns = equity.pct_change().fillna(0)


def generate_random_portfolios(stock_prices, portfolio, mask=None, weighting_factor=None,
                               scaling_factor=None, target=1, balance=None, fee_rate=0, n=100,
                               random_seed=None):
    """Creates `n` random portfolios, replicating portfolio `picks`.

    Parameters
    ----------
    stock_prices : pd.DataFrame
        Prices, representing stock universe.
    portfolio : Portfolio
        Portfolio to be replicated by random ones.
    mask : pd.DataFrame, optional
        Matrix to filter stock universe. By default, stocks which are not
        available for trading are excluded, so there is no necessity to pass
        this mask, if specific filter is not used.
    weighting_factor : Factor, optional
        Factor, weighting positions. If not given, just weigh equally.
    scaling_factor : Factor, optional
        Factor to scale (leverage) positions weights.
    target : array_like, default=1
        Target of `scaling_factor`.
    balance : int or float, optional
        Initial balance of the portfolio.
    fee_rate : int or float, default=0
        Indicative commission rate for every deal.
    n : int > 0, default=100
        How many random portfolios to be created.
    random_seed : int, optional
        Random seed to make random deterministic.

    Returns
    -------
    list of Portfolio
        Generated random portfolios.
    """

    np.random.seed(random_seed)

    picks = portfolio.picks.astype(float)
    picks[stock_prices.isna()] = np.nan
    if mask is not None:
        picks[~mask] = np.nan

    portfolios = []
    for _ in range(n):
        random_portfolio = Portfolio('random')
        random_portfolio.pick_randomly(picks)
        if weighting_factor is not None:
            random_portfolio.weigh_by_factor(weighting_factor)
        else:
            random_portfolio.weigh_equally()
        if scaling_factor is not None:
            random_portfolio.scale_by_factor(scaling_factor, target)
        random_portfolio.allocate(stock_prices, balance, fee_rate)

        portfolios.append(random_portfolio)

    return portfolios
