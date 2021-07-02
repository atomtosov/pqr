"""
This module contains interfaces for different kinds of portfolios:
    * interface for simple factor portfolio, investing by picking stocks by
    some interval of factor values
    * interface for Winners-Minus-Losers Portfolio (WML), investing in stocks,
    invested by winners-portfolio, and selling those invested by
    losers-portfolio
"""

from .iportfolio import IPortfolio
from .iwmlportfolio import IWMLPortfolio
