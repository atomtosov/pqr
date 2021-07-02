"""
This module contains 2 kinds of portfolios:
    * ordinary factor portfolio (Portfolio) - which invest in stocks, picked by
    some factor and some interval (only-long)
    * winners-minus-losers portfolio (WMLPortfolio) - which invest in stocks,
    picked by portfolio-winner, and sell short picks of portfolio-loser
"""

from .portfolio import Portfolio
from .wmlportfolio import WMLPortfolio
