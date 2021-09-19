"""This module contains presets to visualize analysis of portfolios performance and compare them. These presets 
are supposed to be used in Jupyter Notebook to fastly estimate which portfolios are really profitable and which of
them do not deserve attention.
"""

from IPython.display import display
import pandas as pd
import matplotlib.pyplot as plt

from .metrics import summary
from .plotting import plot_compound_returns

__all__ = [
    'summary_tear_sheet',
]


def summary_tear_sheet(portfolios, benchmark):
    """Shows the performance assessment of `portfolios`.

    For now:

    * shows summary stats table
    * plots compound returns

    Parameters
    ----------
    portfolios : sequence of Portfolio
        Portfolios, included into the factor model.
    benchmark : Portfolio or Benchmark
        Benchmark to compute some metrics.
    """

    stats = pd.DataFrame([summary(p, benchmark) for p in portfolios]).T.round(2)
    display(stats)

    plot_compound_returns(portfolios, benchmark)
    plt.show()
