from abc import abstractmethod
from collections import namedtuple
from typing import Union, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.regression.rolling import RollingOLS

from pqr.benchmarks.interfaces import IBenchmark


Alpha = namedtuple('Alpha', ['coef', 'p_value'])
Beta = namedtuple('Beta', ['coef', 'p_value'])


class BasePortfolio:
    """
    Abstract base class for portfolios of stocks.
    """

    positions: pd.DataFrame
    returns: pd.Series
    benchmark: Optional[IBenchmark]
    shift: int
    cumulative_returns: pd.Series
    total_return: Union[int, float]

    _freq_alias_to_num = {
            # yearly
            'BA': 1,
            'A': 1,
            # quarterly
            'BQ': 4,
            'Q': 4,
            # monthly
            'BM': 12,
            'M': 12,
            # weekly
            'W': 52,
            # daily
            'B': 252,
            'D': 252,
        }

    def __repr__(self) -> str:
        """
        Dunder/Magic method for fancy printing BasePortfolio object in console.
        """
        return f'{self.__class__.__name__}({self._name})'

    @property
    @abstractmethod
    def positions(self) -> pd.DataFrame:
        """
        pd.DataFrame : Dataframe portfolio's positions in each period.
        """

    @property
    @abstractmethod
    def returns(self) -> pd.Series:
        """
        pd.Series: Series of period-to-period returns of portfolio.
        """

    @property
    @abstractmethod
    def benchmark(self) -> Optional[IBenchmark]:
        """
        IBenchmark : Benchmark, which is needed to compare performance of
        portfolio and calculate some metrics.
        """

    @property
    @abstractmethod
    def shift(self) -> int:
        """
        int : Non-tradable period of portfolio (when positions cannot be opened
        because of lack of information).
        """

    @property
    @abstractmethod
    def _name(self) -> str:
        """
        str: Name of portfolio.
        """

    def _get_periodicity(self) -> int:
        """
        Calculate periodicity of positions to calculate some periodical
        metrics (or annualize them).

        Returns
        -------
        int
            Number of periods in 1 year.
        """

        freq_num = self._freq_alias_to_num.get(
            self.positions.index.freq.freqstr,
            None
        )
        if freq_num is None:
            raise ValueError('periodicity of given data cannot be defined, '
                             'try to resample data')
        return freq_num

    @property
    def cumulative_returns(self) -> pd.Series:
        """
        pd.Series : Series of cumulative returns of portfolio.
        """

        return (1 + self.returns).cumprod() - 1

    @property
    def total_return(self) -> Union[int, float]:
        """
        int, float : Total return of portfolio (last element of cumulative
        return series).
        """

        return self.cumulative_returns[-1]

    def calc_alpha(self,
                   rolling: bool = False) -> Union[Alpha, pd.Series]:
        """
        Calculate Alpha of portfolio.

        Alpha is the a in the estimated regression into benchmark returns:
            r_portfolio = a + b * r_benchmark + e, e - stochastic error

        Parameters
        ----------
        rolling : bool
            If True, returns series of rolling yearly alphas, else returns
            alpha over the all time of investing.

        Returns
        -------
        Alpha, pd.Series
            If not rolling, returns namedtuple Alpha with fields "coef" and
            "p-value", else returns pd.Series of rolling yearly alphas.
        """

        returns = self.returns.values[self.shift + 1:]
        benchmark_returns = self.benchmark.returns.values[self.shift + 1:]
        x = sm.add_constant(np.nan_to_num(benchmark_returns))

        if rolling:
            est = RollingOLS(returns, x, window=self._get_periodicity()).fit()
            return pd.Series(
                est.params[:, 0],
                index=self.returns.index[self.shift + 1:]
            )

        est = OLS(returns, x).fit()
        return Alpha(
            coef=est.params[0],
            p_value=est.pvalues[0]
        )

    def calc_beta(self,
                  rolling: bool = False) -> Union[Beta, pd.Series]:
        """
        Calculate Beta of portfolio.

        Beta is the b in the estimated regression into benchmark returns:
            r_portfolio = a + b * r_benchmark + e, e - stochastic error

        Parameters
        ----------
        rolling : bool
            If True, returns series of rolling yearly betas, else returns
            beta over the all time of investing.

        Returns
        -------
        Beta, pd.Series
            If not rolling, returns namedtuple Beta with fields "coef" and
            "p-value", else returns pd.Series of rolling yearly betas.
        """

        returns = self.returns.values[self.shift + 1:]
        benchmark_returns = self.benchmark.returns.values[self.shift + 1:]
        x = sm.add_constant(np.nan_to_num(benchmark_returns))

        if rolling:
            est = RollingOLS(returns, x,
                             window=self._get_periodicity()).fit()
            return pd.Series(
                est.params[:, 1],
                index=self.returns.index[self.shift + 1:]
            )

        est = OLS(returns, x).fit()
        return Beta(
            coef=est.params[1],
            p_value=est.pvalues[1]
        )

    def calc_sharpe(self,
                    rolling: bool = False) -> Union[int, float, pd.Series]:
        """
        Calculate annualized Sharpe Ratio of portfolio.

        Sharpe Ratio is the ratio of mean return and standard deviation of
        returns:
            SR = r_mean / std(r) * A, A - annualizing coefficient, equals to
            sqrt(number of periods in a year) (e.g. for monthly data
            A = sqrt(12))

        Parameters
        ----------
        rolling : bool
            If True, returns series of rolling yearly sharpe ratios, else
            returns alpha over the all time of investing.

        Returns
        -------
        int, float, pd.Series
            If not rolling, returns only one number - Sharpe Ratio, else
            returns pd.Series of rolling yearly sharpe ratios.
        """

        if rolling:
            return self.returns.rolling(self._get_periodicity()).mean() \
                   / self.returns.rolling(self._get_periodicity()).std() \
                   * np.sqrt(self._get_periodicity())

        return self.returns.mean() / self.returns.std() * \
            np.sqrt(self._get_periodicity())

    def calc_mean_return(self,
                         rolling: bool = False) -> Union[int, float,
                                                         pd.Series]:
        """
        Calculate mean 1-period return of portfolio.

        Simply mean 1-period return of n portfolio returns:
            MR = sum(r_i)_i=0,n / n

        Parameters
        ----------
        rolling : bool
            If True, returns series of rolling yearly mean returns, else
            returns mean return over the all time of investing.

        Returns
        -------
        int, float, pd.Series
            If not rolling, returns only one number - mean 1-period return,
            else returns pd.Series of rolling yearly mean returns.
        """

        if rolling:
            return self.returns.rolling(self._get_periodicity()).mean()

        return self.returns.values.mean()

    def calc_excessive_return(self,
                              rolling: bool = False) -> Union[int, float,
                                                              pd.Series]:
        """
        Calculate excessive return of portfolio.

        Excessive return is the average difference of portfolio returns and
        benchmark returns for the same periods:
            ER = mean(r_p) - mean(r_b)

        Parameters
        ----------
        rolling : bool
            If True, returns series of rolling yearly excessive returns, else
            returns excessive return over the all time of investing.

        Returns
        -------
        int, float, pd.Series
            If not rolling, returns only one number - mean excessive return,
            else returns pd.Series of rolling yearly mean excessive returns.
        """

        if rolling:
            return self.calc_mean_return(rolling=True) \
                   - self.benchmark.returns.rolling(self._get_periodicity())\
                       .mean()

        return self.calc_mean_return() - np.nanmean(self.benchmark.returns)

    def calc_volatility(self,
                        rolling: bool = False) -> Union[int, float,
                                                        pd.Series]:
        """
        Calculate volatility (standard deviation of 1-period returns) of
        portfolio.

        Volatility is measured as simple standard deviation of 1-period
        portfolio returns:
            V = sqrt(sum((r_i - r_mean)^2)_i=0,n / n)

        Parameters
        ----------
        rolling : bool
            If True, returns series of rolling yearly volatility, else
            returns volatility over the all time of investing.

        Returns
        -------
        int, float, pd.Series
            If not rolling, returns only one number - volatility, else
            returns pd.Series of rolling yearly volatility.
        """

        if rolling:
            return self.returns.rolling(self._get_periodicity()).std()

        return self.returns.values.std()

    def calc_benchmark_correlation(self,
                                   rolling: bool = False) -> Union[int, float,
                                                                   pd.Series]:
        """
        Calculate correlation of portfolio returns and benchmark returns.

        Simple correlation of 2 series - series of portfolio returns ans series
        of benchmark returns:
            Corr = cov(r_p, r_b) / (Var(r_p) * Var(r_b))

        Parameters
        ----------
        rolling : bool
            If True, returns series of rolling yearly correlations, else
            returns correlation over the all time of investing.

        Returns
        -------
        int, float, pd.Series
            If not rolling, returns only one number - correlation, else
            returns pd.Series of rolling yearly correlations.
        """

        if rolling:
            return self.returns.rolling(self._get_periodicity())\
                .corr(self.benchmark.returns)

        return self.returns.corr(self.benchmark.returns)

    def calc_profitable_periods(self,
                                rolling: bool = False) -> Union[int, float,
                                                                pd.Series]:
        """
        Calculate share of profitable periods of portfolio.

        Just ratio of number of profitable periods (return > 0) and the total
        number of periods:
            SPR = sum[r_i > 0]_i=0,n / n

        Parameters
        ----------
        rolling : bool
            If True, returns series of rolling yearly share of profitable
            periods, else returns share of profitable periods over the all time
            of investing.

        Returns
        -------
        int, float, pd.Series
            If not rolling, returns only one number - share of profitable
            periods, else returns pd.Series of rolling yearly shares.
        """

        if rolling:
            return (self.returns > 0).rolling(self._get_periodicity()).sum() \
                   / self._get_periodicity()

        return (self.returns.values > 0).sum() / np.size(self.returns.values)

    def calc_max_drawdown(self,
                          rolling: bool = False) -> Union[int, float,
                                                          pd.Series]:
        """
        Calculate maximum drawdown of portfolio.

        The biggest absolute drawdown of portfolio cumulative returns:
            MD = min(r_i - max(r_j, j=0,i-1), i=0,n)

        Parameters
        ----------
        rolling : bool
            If True, returns series of rolling yearly maximum drawdons, else
            returns maximum drawdown over the all time of investing.

        Returns
        -------
        int, float, pd.Series
            If not rolling, returns only one number - maximum drawdown, else
            returns pd.Series of rolling yearly maximum drawdowns.
        """

        cum_returns = self.cumulative_returns

        if rolling:
            return cum_returns.rolling(self._get_periodicity()).apply(
                lambda x: (x - x.cummax()).min()
            )

        return (cum_returns - cum_returns.cummax()).min()

    @property
    def stats(self) -> Dict[str, Union[int, float]]:
        """
        dict[str, int or float] : Summary statistic of portfolio through the
        all investing period.
        """

        return {
            'Alpha, %': self.calc_alpha().coef * 100,
            'Alpha p-value': self.calc_alpha().p_value,
            'Beta': self.calc_beta().coef,
            'Beta p-value': self.calc_beta().p_value,
            'Sharpe Ratio': self.calc_sharpe(),
            'Mean Return, %': self.calc_mean_return() * 100,
            'Excessive Return, %': self.calc_excessive_return() * 100,
            'Total Return, %': self.total_return * 100,
            'Volatility, %': self.calc_volatility() * 100,
            'Benchmark Correlation': self.calc_benchmark_correlation(),
            'Profitable Periods, %': self.calc_profitable_periods() * 100,
            'Maximum Drawdown, %': self.calc_max_drawdown() * 100,
        }

    def plot_cumulative_returns(self, add_benchmark: bool = True) -> None:
        """
        Plot cumulative returns of portfolio.

        Parameters
        ----------
        add_benchmark : bool
            Whether to add benchmark returns on the graph too, or not.
        """

        plt.plot(self.returns.index,
                 self.cumulative_returns,
                 label=repr(self))
        if add_benchmark and self.benchmark is not None:
            self.benchmark.plot_cumulative_returns(self.shift)

    def plot_rolling_alpha(self) -> None:
        """
        Plot rolling yearly Alpha of portfolio.
        """

        plt.plot(self.calc_alpha(True),
                 label='Rolling Alpha')

    def plot_rolling_beta(self) -> None:
        """
        Plot rolling yearly Beta of portfolio.
        """

        plt.plot(self.calc_beta(True),
                 label='Rolling Beta')

    def plot_rolling_sharpe(self) -> None:
        """
        Plot rolling yearly Sharpe Ratio of portfolio.
        """

        plt.plot(self.calc_sharpe(True),
                 label='Rolling Sharpe Ratio')

    def plot_rolling_mean_return(self) -> None:
        """
        Plot rolling yearly Mean Return of portfolio.
        """

        plt.plot(self.calc_mean_return(True),
                 label='Rolling Mean Return')

    def plot_rolling_excessive_return(self) -> None:
        """
        Plot rolling yearly Excessive Return of portfolio.
        """

        plt.plot(self.calc_excessive_return(True),
                 label='Rolling Excessive Return')

    def plot_rolling_volatility(self) -> None:
        """
        Plot rolling yearly Volatility of portfolio returns.
        """

        plt.plot(self.calc_volatility(True),
                 label='Rolling Volatility')

    def plot_rolling_benchmark_correlation(self) -> None:
        """
        Plot rolling yearly correlation of portfolio returns and benchmark
        returns.
        """

        plt.plot(self.calc_benchmark_correlation(True),
                 label='Rolling Benchmark Correlation')

    def plot_rolling_profitable_periods(self) -> None:
        """
        Plot rolling yearly share of profitable periods of portfolio.
        """

        plt.plot(self.calc_profitable_periods(True),
                 label='Rolling Profitable Periods')

    def plot_rolling_max_drawdown(self) -> None:
        """
        Plot rolling yearly Maximum Drawdown of portfolio.
        """

        plt.plot(self.calc_max_drawdown(True),
                 label='Rolling Maximum Drawdown')
