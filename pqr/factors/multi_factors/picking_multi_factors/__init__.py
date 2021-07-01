"""
This module contains only picking multi-factors. There are different approaches
to combine some factors into multi-factors to pick stocks, but this module
implements for now only 3 of them:
    * simple weigh given factors (WeighMultiFactor)
    * intercept choices of each factor (InterceptMultiFactor)
    * iteratively sort stock universe by each factor (NSortMultiFactor)
"""

from .weighmultifactor import WeighMultiFactor
from .interceptmultifactor import InterceptMultiFactor
from .nsortmultifactor import NSortMultiFactor
