__all__ = [
    "display_dashboard",
    "plot_chart",
    "show_table",
    "fancy_format",
]

from typing import (
    Callable,
    Sequence,
    Optional,
    Dict,
)

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display


def display_dashboard(
        portfolios: Sequence[pd.DataFrame],
        items: Sequence[Callable[[Sequence[pd.DataFrame]], None]],
) -> None:
    for item in items:
        item(portfolios)


def plot_chart(
        portfolios: Sequence[pd.DataFrame],
        metrics: Dict[str, Callable[[pd.DataFrame], pd.Series]],
        benchmark: Optional[pd.Series] = None,
        **kwargs,
) -> None:
    plt.figure(**kwargs)

    for name, metric in metrics.items():
        for portfolio in portfolios:
            plt.plot(metric(portfolio), label=portfolio.index.name)

        if benchmark is not None:
            starts_from = min(portfolio.index[0] for portfolio in portfolios)
            plt.plot(
                metric(benchmark[starts_from:]),
                label=benchmark.index.name,
                color="gray", alpha=0.8,
            )

    metric_names = ', '.join(metrics.keys())
    plt.title(f"Portfolios {metric_names}")
    plt.xlabel("Date")
    plt.ylabel(metric_names)
    plt.legend()
    plt.grid()

    plt.show()


def show_table(
        portfolios: Sequence[pd.DataFrame],
        metrics: Dict[str, Callable[[pd.DataFrame], str]],
) -> None:
    metrics_table = {}
    for name, metric in metrics.items():
        metrics_table[name] = [
            metric(portfolio)
            for portfolio in portfolios
        ]

    display(
        pd.DataFrame(
            metrics_table,
            index=[portfolio.index.name for portfolio in portfolios]
        ).T
    )


def fancy_format(
        metric: Callable,
        *,
        multiplier: float = 1.0,
        precision: int = 2,
        **kwargs,
) -> Callable[[pd.DataFrame], str]:
    def fancy_metric(portfolio):
        metric_value = metric(portfolio, **kwargs)

        if isinstance(metric_value, tuple):  # if statistics
            coef, t_stat, p_value = metric_value
            return "{coef}{stars} ({t_stat})".format(
                coef=format(coef * multiplier, f".{precision}f"),
                stars="*" * _count_stars(p_value),
                t_stat=format(t_stat, f".{precision}f"),
            )
        else:
            return format(
                metric_value * multiplier,
                f".{precision}f"
            )

    return fancy_metric


def _count_stars(p_value: float) -> int:
    if p_value < 0.01:
        return 3
    elif p_value < 0.05:
        return 2
    elif p_value < 0.1:
        return 1
    else:
        return 0
