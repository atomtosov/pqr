"""
    Module, providing multi-factors:
        picking factors (simply factors): weigh, intercept and nsort,
        filtering factors,
        weighting factors.
"""

from .weighmultifactor import WeighMultiFactor
from .interceptmultifactor import InterceptMultiFactor
from .nsortmultifactor import NSortMultiFactor

from .filteringmultifactor import FilteringMultiFactor
from .weightingmultifactor import WeightingMultiFactor
