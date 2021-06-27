from typing import Optional
import pandas as pd

from .baseportfolio import BasePortfolio
from .interfaces import IRelativeInvest
from pqr.benchmarks import BaseBenchmark


class WMLPortfolio(BasePortfolio, IRelativeInvest):
    def __init__(self):
        self._positions = pd.DataFrame()
        self._returns = pd.Series()
        self._benchmark = None
        self._shift = 0

        self._winners = None
        self._losers = None

    def invest(self,
               winners: BasePortfolio,
               losers: BasePortfolio,
               benchmark: BaseBenchmark) -> None:
        if isinstance(benchmark, BaseBenchmark) or benchmark is None:
            self._benchmark = benchmark
        else:
            raise TypeError('benchmark must implement IBenchmark')

        self._winners = winners
        self._losers = losers

    @property
    def positions(self) -> pd.DataFrame:
        return self.winners.positions - self.losers.positions

    @property
    def returns(self) -> pd.Series:
        return self.winners.returns - self.losers.returns

    @property
    def benchmark(self) -> Optional['BaseBenchmark']:
        return self._benchmark

    @property
    def shift(self) -> int:
        return self._shift

    @property
    def _name(self) -> str:
        return ''

    @property
    def winners(self) -> BasePortfolio:
        return self._winners

    @property
    def losers(self) -> BasePortfolio:
        return self._losers
