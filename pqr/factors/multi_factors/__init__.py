"""
    Module, providing multi-factors:
        picking factors (simply factors): weigh, intercept and nsort,
        filtering factors,
        weighting factors.
"""

from .picking_multi_factors import (
    WeighMultiFactor,
    InterceptMultiFactor,
    NSortMultiFactor
)

from .filteringmultifactor import FilteringMultiFactor
from .weightingmultifactor import WeightingMultiFactor
