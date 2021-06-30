"""
This module contains all interfaces for different usage cases of factors:
    * picking - factor is used to pick some stocks from all stock universe by
    some interval (e.g. Quantile(0, 1/2)
    * filtering - factor is used to filter stock universe by some thresholds
    (e.g. to delete from stock universe low-liquid stocks with avg daily
    trading volume < 1 000 000$)
    * weighting - factor is used to weigh positions by factor values (e.g.
    by market capitalization)

Interfaces are used instead of base classes, because they are implemented not
only by factors, but also by multi-factors and dummy-factors (e.g. NoFilter).
"""

from .ipicking import IPicking
from .ifiltering import IFiltering
from .iweighting import IWeighting
