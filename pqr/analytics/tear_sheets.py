"""This module contains presets to visualize analysis of portfolios performance and compare them. These presets 
are supposed to be used in Jupyter Notebook to fastly estimate which portfolios are really profitable and which of
them do not deserve attention.
"""

from IPython.display import display
import pandas as pd
import matplotlib.pyplot as plt

from .metrics import *
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

    stats = []
    for portfolio in portfolios:
        portfolio_stats = pd.Series(
            {
            'Total Return, %': total_return(portfolio.returns) * 100,
            'CAGR, %': cagr(portfolio.returns) * 100,
            'Volatility, %': volatility(portfolio.returns) * 100,
            'Win Rate, %': win_rate(portfolio.returns) * 100,
            'Maximum Drawdown, %': max_drawdown(portfolio.returns) * 100,
            'VaR, %': value_at_risk(portfolio.returns) * 100,
            'Expected Tail Loss, %': expected_tail_loss(portfolio.returns) * 100,
            'Rachev Ratio': rachev_ratio(portfolio.returns),
            'Calmar Ratio': calmar_ratio(portfolio.returns),
            'Sharpe Ratio': sharpe_ratio(portfolio.returns),
            'Omega Ratio': omega_ratio(portfolio.returns),
            'Sortino Ratio': sortino_ratio(portfolio.returns),
            'Benchmark Correlation': benchmark_correlation(portfolio.returns, benchmark.returns),
            'Turnover, %': turnover(portfolio.positions),
        },
        name=portfolio.name
        ).round(2).astype(str)

        # add stars and t-stat 
        for name, metric in zip(['Mean Return, %', 'Mean Excess Return, %', 'Alpha, %', 'Beta'],
                                [mean_return, mean_excess_return, alpha, beta]):
            if name != 'Mean Return, %':
                metric_values = metric(portfolio.returns, benchmark.returns)
            else:
                metric_values = metric(portfolio.returns)
                
            portfolio_stats[name] = '{:.2f}{}\n({:.2f})'.format(
                metric_values.value * (100 if '%' in name else 1), 
                _stars(metric_values.p_value),
                metric_values.t_stat
            )

        stats.append(portfolio_stats)
    
    display(pd.DataFrame(stats).T.style.set_properties(**{'white-space': 'pre-wrap'}))

    plt.yscale('symlog')
    plot_compound_returns(portfolios, benchmark)
    plt.show()


def _stars(p_value):
    if p_value < 0.01:
        return '***'
    elif p_value < 0.05:
        return '**'
    elif p_value < 0.1:
        return '*'
    else:
        return ''
