"""
This module contains only multi-factor implementations of factors.
Multi-factor is factor, which can be represented by more than 1 matrix in
general (but of course it might be in some cases (e.g. simply weigh factors)).
Multi-factors are created to make more complicated models by including more
factors into the processes of picking, filtering and weighting stocks in
portfolio.
"""

from .picking_multi_factors import (
    WeighMultiFactor,
    InterceptMultiFactor,
    NSortMultiFactor
)

from .filteringmultifactor import FilteringMultiFactor
from .weightingmultifactor import WeightingMultiFactor
