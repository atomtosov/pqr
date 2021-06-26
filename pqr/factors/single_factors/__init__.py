"""
    Module, providing single-factors:
        picking factors (simply factors),
        filtering factors,
        weighting factors.
"""

from .singlefactor import SingleFactor

from .factor import Factor
from .filteringfactor import FilteringFactor, NoFilter
from .weightingfactor import WeightingFactor, EqualWeights
