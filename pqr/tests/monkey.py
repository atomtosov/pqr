__all__ = [
    "monkey_test",
]

from typing import (
    Callable,
    Optional,
)

import numpy as np
import pandas as pd

from pqr.core import backtest_portfolio
from pqr.utils import (
    estimate_holding,
    align,
    partial,
    longs_from_portfolio,
    shorts_from_portfolio,
)


def monkey_test(
        portfolio: pd.DataFrame,
        prices: pd.DataFrame,
        universe: pd.DataFrame,
        allocation: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
        n: int = 100,
        holding: Optional[int] = None,
        random_seed: Optional[int] = None
) -> pd.Series:
    if holding is None:
        holding = estimate_holding(portfolio)

    rng = np.random.default_rng(random_seed)

    random_portfolio_factory = get_random_portfolio_factory(
        base_portfolio=portfolio,
        prices=prices,
        universe=universe,
        allocation=allocation,
        holding=holding,
        rng=rng,
    )

    random_returns = np.array([
        random_portfolio_factory()["returns"] for _ in range(n)
    ]).T

    outperform = random_returns <= portfolio["returns"].to_numpy()[:, np.newaxis]

    monkey_est = pd.Series(
        np.nanmean(outperform, axis=1)[1:],
        index=portfolio.returns.index.copy()[1:]
    )
    monkey_est.index.name = "Monkey"

    return monkey_est


def get_random_portfolio_factory(
        base_portfolio: pd.DataFrame,
        prices: pd.DataFrame,
        universe: pd.DataFrame,
        allocation: Callable[[pd.DataFrame], pd.DataFrame],
        holding: int,
        rng: np.random.Generator,
) -> Callable[[], pd.DataFrame]:
    longs, shorts = longs_from_portfolio(base_portfolio), shorts_from_portfolio(base_portfolio)
    long_random_strategy = partial(
        pick_randomly,
        universe=universe,
        holdings_number=longs.sum(axis=1),
        holding=holding,
        rng=rng,
    )
    short_random_strategy = partial(
        pick_randomly,
        universe=universe,
        holdings_number=shorts.sum(axis=1),
        holding=holding,
        rng=rng,
    )

    return lambda: backtest_portfolio(
        prices=prices,
        longs=long_random_strategy(),
        shorts=short_random_strategy(),
        allocation=allocation,
        name="Random"
    )


def pick_randomly(
        universe: pd.DataFrame,
        holdings_number: pd.Series,
        holding: int,
        rng: np.random.Generator
) -> pd.DataFrame:
    universe, holdings_number = align(universe, holdings_number)
    universe_array = universe.to_numpy()
    holdings_number_array = holdings_number.to_numpy()

    choices = np.zeros_like(universe_array, dtype=bool)
    stop = len(choices)
    for i in range(0, stop, holding):
        tradable = np.where(universe_array[i])[0]
        choice = rng.choice(
            tradable,
            size=holdings_number_array[i],
            replace=False
        )
        choices[i:min(stop, i + holding), choice] = True

    return pd.DataFrame(
        choices,
        index=universe.index.copy(),
        columns=universe.columns.copy(),
    )
