"""
This module contains intervals for picking stocks by the diapason of factor
values. Picking stock process can be done by:
    * quantiles - boarders are percentiles to include
    * thresholds - boarders are simple limits for thresholds
    * top - boarders are places in sorted factor values in each period
"""

from .interval import Interval

from .quantiles import Quantiles
from .thresholds import Thresholds
from .top import Top
