from __future__ import annotations

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
    q = np.linspace(0.0, 1.0, quantiles + 1)
    factor_quantiles = factor.quantile(q).to_numpy()
    factor_values = factor.values.to_numpy()

    portfolios = [
        Portfolio("Winners").pick(
            long=Universe(
                array_to_alike_df_or_series(
                    (factor_quantiles[:, [-2]] <= factor_values) &
                    (factor_values <= factor_quantiles[:, [-1]])
                    if better == "more" else
                    (factor_quantiles[:, [0]] <= factor_values) &
                    (factor_values <= factor_quantiles[:, [1]]),
                    factor.values
                )
            )
        ),
        *[
            Portfolio(f"Neutral {i}").pick(
                long=Universe(
                    array_to_alike_df_or_series(
                        (factor_quantiles[:, [quantiles - i - 1]] <= factor_values) &
                        (factor_values <= factor_quantiles[:, [quantiles - i]])
                        if better == "more" else
                        (factor_quantiles[:, [1]] <= factor_values) &
                        (factor_values <= factor_quantiles[:, [i + 1]]),
                        factor.values
                    )
                )
            )
            for i in range(1, quantiles - 1)
        ],
        Portfolio("Losers").pick(
            long=Universe(
                array_to_alike_df_or_series(
                    (factor_quantiles[:, [0]] <= factor_values) &
                    (factor_values <= factor_quantiles[:, [1]])
                    if better == "more" else
                    (factor_quantiles[:, [-2]] <= factor_values) &
                    (factor_values <= factor_quantiles[:, [-1]]),
                    factor.values
                )
            )
        ),
    ]

    if add_wml:
        portfolios.append(
            Portfolio("WML").pick(
                long=Universe(portfolios[0].picks == 1),
                short=Universe(portfolios[-1].picks == 1)
            )
        )

    for portfolio in portfolios:
        portfolio.weigh(weighting_factor)
        portfolio.allocate(prices, fee)

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
    factor_values = factor.values.to_numpy()

    portfolios = [
        Portfolio("Winners").pick(
            long=Universe(
                array_to_alike_df_or_series(
                    (factor_values >= threshold)
                    if better == "more" else
                    (factor_values <= threshold),
                    factor.values
                )
            )
        ),
        Portfolio("Losers").pick(
            long=Universe(
                array_to_alike_df_or_series(
                    (factor_values <= threshold)
                    if better == "more" else
                    (factor_values >= threshold),
                    factor.values
                )
            )
        ),
    ]

    if add_wml:
        portfolios.append(
            Portfolio("WML").pick(
                long=Universe(portfolios[0].picks == 1),
                short=Universe(portfolios[-1].picks == 1)
            )
        )

    for portfolio in portfolios:
        portfolio.weigh(weighting_factor)
        portfolio.allocate(prices, fee)

    return portfolios


def build_top_portfolios(
        prices: pd.DataFrame,
        factor: Factor,
        better: Literal["more", "less"] = "more",
        weighting_factor: Optional[Factor] = None,
        fee: float = 0.0,
        n: int = 10,
        add_wml: bool = False,
) -> list[Portfolio]:
    factor_top = factor.top([n]).to_numpy()
    factor_bottom = factor.bottom([n]).to_numpy()
    factor_values = factor.values.to_numpy()

    portfolios = [
        Portfolio("Winners").pick(
            long=Universe(
                array_to_alike_df_or_series(
                    (factor_values >= factor_top)
                    if better == "more" else
                    (factor_values <= factor_bottom),
                    factor.values
                )
            )
        ),
        Portfolio("Losers").pick(
            long=Universe(
                array_to_alike_df_or_series(
                    (factor_values <= factor_bottom)
                    if better == "more" else
                    (factor_values >= factor_top),
                    factor.values
                )
            )
        )
    ]

    if add_wml:
        portfolios.append(
            Portfolio("WML").pick(
                long=Universe(portfolios[0].picks == 1),
                short=Universe(portfolios[-1].picks == 1)
            )
        )

    for portfolio in portfolios:
        portfolio.weigh(weighting_factor)
        portfolio.allocate(prices, fee)

    return portfolios


def grid_search(
        prices: pd.DataFrame,
        factor_values: pd.DataFrame,
        params: list[tuple[int, int, int]],
        agg: Callable[[np.ndarray], npt.ArrayLike],
        target: Callable[[Portfolio], float],
        **kwargs,
) -> pd.DataFrame:
    metrics = []
    for looking, lag, holding in params:
        factor = Factor(factor_values)
        factor.look_back(agg, looking)
        factor.lag(lag)
        factor.hold(holding)

        if kwargs.get("quantiles"):
            portfolios = build_quantile_portfolios(prices, factor, **kwargs)
        elif kwargs.get("n"):
            portfolios = build_top_portfolios(prices, factor, **kwargs)
        else:
            portfolios = build_time_series_portfolios(prices, factor, **kwargs)

        metrics.append(
            pd.DataFrame(
                [[target(portfolio) for portfolio in portfolios]],
                index=[(looking, lag, holding)],
                columns=[portfolio.name for portfolio in portfolios]
            )
        )

    return pd.concat(metrics)
