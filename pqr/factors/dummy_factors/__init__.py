"""
This module provides dummy implementations for interfaces. They can be used to
replace expected, but not passed factors, because they implements the same
interfaces. Actually, some of them do nothing, others do the simplest approach
to realise the supposed interface.
"""

from .nofilter import NoFilter
from .equalweights import EqualWeights
