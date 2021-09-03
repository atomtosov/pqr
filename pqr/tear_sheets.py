import pandas as pd

from .metrics import summary
from .plotting import plot_cumulative_returns

__all__ = [
    'summary_tear_sheet',
]


def summary_tear_sheet(portfolios, benchmark):
    """Shows the performance assessment of `portfolios`.

    For now:

    * shows summary stats table
    * plots cumulative returns

    Parameters
    ----------
    portfolios : sequence of Portfolio
        Portfolios, included into the factor model.
    benchmark : Portfolio or Benchmark
        Benchmark to compute some metrics.

    Returns
    -------
    pd.DataFrame
        Table with summary stats.
    """

    stats = pd.DataFrame([summary(p, benchmark) for p in portfolios]).T.round(2)
    plot_cumulative_returns(portfolios, benchmark)
    return stats
