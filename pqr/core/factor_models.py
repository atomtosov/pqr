from __future__ import annotations

from copy import copy
from typing import Literal, Optional, Callable

import numpy as np
import numpy.typing as npt
import pandas as pd

from .factor import Factor
from .portfolio import Portfolio
from .universe import Universe
from ..utils import array_to_alike_df_or_series

__all__ = [
    "build_quantile_portfolios",
    "build_top_portfolios",
    "build_time_series_portfolios",
    "grid_search",
]


def build_quantile_portfolios(
        prices: pd.DataFrame,
        factor: Factor,
        better: Literal["more", "less"] = "more",
        weighting_factor: Optional[Factor] = None,
        fee: float = 0.0,
        quantiles: int = 3,
        add_wml: bool = False,
) -> list[Portfolio]:
    # TODO: refactor that
    q: np.ndarray = np.linspace(0.0, 1.0, quantiles + 1)

    portfolios_names: list[str] = [
        "Losers",
        *[f"Neutral {j}" for j in range(quantiles - 2, 0, -1)],
        "Winners"
    ]
    portfolios: list[Portfolio] = []

    for i in (
            range(quantiles, 0, -1) if better == "more"
            else range(1, quantiles + 1)
    ):
        picks: pd.DataFrame = array_to_alike_df_or_series(
            (factor.quantile(q[i - 1]).to_numpy()[:, np.newaxis] <= factor.values.to_numpy())
            &
            (factor.values.to_numpy() <= factor.quantile(q[i]).to_numpy()[:, np.newaxis]),
            factor.values
        )

        portfolio: Portfolio = Portfolio(portfolios_names.pop())
        portfolio.pick(long=Universe(picks))
        portfolio.weigh(weighting_factor)
        portfolio.allocate(prices, fee)

        portfolios.append(portfolio)

    if add_wml:
        wml: Portfolio = Portfolio("WML")
        wml.pick(
            long=Universe(portfolios[0].picks == 1),
            short=Universe(portfolios[-1].picks == 1)
        )
        wml.weigh(weighting_factor)
        wml.allocate(prices, fee)

        portfolios.append(wml)

    return portfolios


def build_time_series_portfolios(
        prices: pd.DataFrame,
        factor: Factor,
        better: Literal["more", "less"] = "more",
        weighting_factor: Optional[Factor] = None,
        fee: float = 0.0,
        threshold: float = 0.0,
        add_wml: bool = False,
) -> list[Portfolio]:
    winners_picks: pd.DataFrame = array_to_alike_df_or_series(
        threshold >= factor.values.to_numpy(),
        factor.values
    )
    losers_picks: pd.DataFrame = array_to_alike_df_or_series(
        threshold <= factor.values.to_numpy(),
        factor.values
    )
    if better == "more":
        winners_picks, losers_picks = losers_picks, winners_picks

    portfolios: list[Portfolio] = [
        Portfolio("Winners").pick(long=Universe(winners_picks)),
        Portfolio("Losers").pick(long=Universe(losers_picks)),
    ]
    for portfolio in portfolios:
        portfolio.weigh(weighting_factor)
        portfolio.allocate(prices, fee)

    if add_wml:
        wml: Portfolio = Portfolio("WML")
        wml.pick(
            long=Universe(portfolios[0].picks == 1),
            short=Universe(portfolios[-1].picks == 1)
        )
        wml.weigh(weighting_factor)
        wml.allocate(prices, fee)

        portfolios.append(wml)

    return portfolios


def build_top_portfolios(
        prices: pd.DataFrame,
        factor: Factor,
        better: Literal["more", "less"] = "more",
        weighting_factor: Optional[Factor] = None,
        fee: float = 0.0,
        place: int = 10,
        add_wml: bool = False,
) -> list[Portfolio]:
    winners_picks: pd.DataFrame = array_to_alike_df_or_series(
        (factor.top(place).to_numpy()[:, np.newaxis] >= factor.values.to_numpy()),
        factor.values
    )
    losers_picks: pd.DataFrame = array_to_alike_df_or_series(
        (factor.bottom(place).to_numpy()[:, np.newaxis] <= factor.values.to_numpy()),
        factor.values
    )
    if better == "less":
        winners_picks, losers_picks = losers_picks, winners_picks

    portfolios: list[Portfolio] = [
        Portfolio("Winners").pick(Universe(winners_picks)),
        Portfolio("Losers").pick(Universe(losers_picks))
    ]
    for portfolio in portfolios:
        portfolio.weigh(weighting_factor)
        portfolio.allocate(prices, fee)

    if add_wml:
        wml: Portfolio = Portfolio("WML")
        wml.pick(
            long=Universe(portfolios[0].picks == 1),
            short=Universe(portfolios[-1].picks == 1)
        )
        wml.weigh(weighting_factor)
        wml.allocate(prices, fee)

        portfolios.append(wml)

    return portfolios


def grid_search(
        prices: pd.DataFrame,
        factor: Factor,
        params: list[tuple[int, int, int]],
        agg: Callable[[np.ndarray], npt.ArrayLike],
        target: Callable[[Portfolio], float],
        **kwargs,
) -> pd.DataFrame:
    metrics = []
    for looking, lag, holding in params:
        factor_ = copy(factor)
        factor_.look_back(agg, looking)
        factor_.lag(lag)
        factor_.hold(holding)

        if kwargs.get("quantiles"):
            portfolios = build_quantile_portfolios(prices, factor_, **kwargs)
        elif kwargs.get("place"):
            portfolios = build_top_portfolios(prices, factor_, **kwargs)
        else:
            portfolios = build_time_series_portfolios(prices, factor_, **kwargs)

        metric = pd.DataFrame(
            [[target(portfolio) for portfolio in portfolios]],
            index=[(looking, lag, holding)],
            columns=[portfolio.name for portfolio in portfolios]
        )

        metrics.append(metric)

    return pd.concat(metrics)
