"""
This module contains only single-factor implementations of factors.
Single-factor is factor, which can be represented only by one matrix. So, this
module provides simplest factors.
"""

from .singlefactor import SingleFactor

from .pickingfactor import PickingFactor
from .filteringfactor import FilteringFactor
from .weightingfactor import WeightingFactor
