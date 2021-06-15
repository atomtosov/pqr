import numpy as np

from .baseportfolio import BasePortfolio
from pqr.benchmarks import BaseBenchmark


class WMLPortfolio(BasePortfolio):
    def construct(self,
                  winners: BasePortfolio,
                  losers: BasePortfolio,
                  benchmark: BaseBenchmark) -> None:
        self.index = np.array(winners.index)
        self._positions = winners.positions - losers.positions
        self._returns = (winners.returns - losers.returns)[:, np.newaxis]
        if isinstance(benchmark, BaseBenchmark):
            self._benchmark = benchmark
