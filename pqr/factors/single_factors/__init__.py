"""
    Module, providing single-factors:
        picking factors (simply factors),
        filtering factors,
        weighting factors.
"""

from .singlefactor import SingleFactor

from .pickingfactor import PickingFactor
from .filteringfactor import FilteringFactor, NoFilter
from .weightingfactor import WeightingFactor, EqualWeights
