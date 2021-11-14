from dataclasses import dataclass

from scipy.stats import ttest_1samp

from pqr.core import Portfolio


@dataclass
class TTest:
    h0: float = 0.0
    alternative: str = "greater"

    def __call__(self, portfolio: Portfolio):
        return ttest_1samp(portfolio.returns, self.h0, alternative=self.alternative)
