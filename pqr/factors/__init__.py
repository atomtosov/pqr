"""
    Module, providing factors (single and multi):
        picking factors (simply factors),
        filtering factors,
        weighting factors.
"""

from .dummy_factors import NoFilter, EqualWeights

from .single_factors import PickingFactor, FilteringFactor, WeightingFactor

from .multi_factors import (
    WeighMultiFactor, InterceptMultiFactor, NSortMultiFactor,
    FilteringMultiFactor,
    WeightingMultiFactor
)
