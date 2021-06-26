"""
    Module, providing factors (single and multi):
        picking factors (simply factors),
        filtering factors,
        weighting factors.
"""

from .single_factors import (
    Factor,
    FilteringFactor, NoFilter,
    WeightingFactor, EqualWeights
)

from .multi_factors import (
    WeighMultiFactor, InterceptMultiFactor, NSortMultiFactor,
    FilteringMultiFactor,
    WeightingMultiFactor
)
