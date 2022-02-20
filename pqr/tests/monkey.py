from __future__ import annotations

__all__ = [
    "monkey_test",
]

from typing import (
    Callable,
    Optional,
    Generator,
)

import numpy as np
import pandas as pd

from pqr.core import Portfolio
from pqr.utils import align


def monkey_test(
        base_portfolio: Portfolio,
        universe: pd.DataFrame,
        allocator: Callable[[pd.DataFrame], pd.DataFrame],
        calculator: Callable[[pd.DataFrame], pd.Series],
        n: int = 100,
        random_seed: Optional[int] = None,
) -> pd.Series:
    random_portfolio_factory = get_random_portfolio_factory(
        base_portfolio=base_portfolio,
        calculator=calculator,
        universe=universe,
        allocator=allocator,
        rng=np.random.default_rng(random_seed),
    )
    random_returns = np.array([
        next(random_portfolio_factory).returns for _ in range(n)
    ]).T

    outperform = random_returns <= base_portfolio.returns.to_numpy()[:, np.newaxis]

    monkey_est = pd.Series(
        np.nanmean(outperform, axis=1)[1:],
        index=base_portfolio.returns.index.copy()[1:]
    )
    monkey_est.name = "Monkey"

    return monkey_est


def get_random_portfolio_factory(
        base_portfolio: Portfolio,
        universe: pd.DataFrame,
        calculator: Callable[[pd.DataFrame], pd.Series],
        allocator: Callable[[pd.DataFrame], pd.DataFrame],
        rng: np.random.Generator,
) -> Generator[Portfolio]:
    longs = base_portfolio.get_long_picks()
    shorts = base_portfolio.get_short_picks()
    long_holdings = longs.sum(axis=1)
    short_holdings = shorts.sum(axis=1)

    while True:
        yield Portfolio.backtest(
            calculator=calculator,
            longs=pick_randomly(
                universe=universe,
                holdings_number=long_holdings,
                rng=rng,
            ),
            shorts=pick_randomly(
                universe=universe,
                holdings_number=short_holdings,
                rng=rng,
            ),
            allocator=allocator,
            name="Random",
        )


def pick_randomly(
        universe: pd.DataFrame,
        holdings_number: pd.Series,
        rng: np.random.Generator,
) -> pd.DataFrame:
    universe, holdings_number = align(universe, holdings_number)
    universe_array = universe.to_numpy()
    holdings_number_array = holdings_number.to_numpy()

    choices = np.zeros_like(universe_array, dtype=bool)
    for i in range(0, len(choices)):
        tradable = np.where(universe_array[i])[0]
        choice = rng.choice(
            tradable,
            size=holdings_number_array[i],
            replace=False,
        )
        choices[i, choice] = True

    return pd.DataFrame(
        choices,
        index=universe.index.copy(),
        columns=universe.columns.copy(),
    )
