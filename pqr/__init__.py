"""Provides:
  1. Toolkit for testing factor strategies
  2. A lot of financial metrics to assess portfolios performance
  3. Fancy visualization of results

Source: https://github.com/atomtosov/pqr/

Affiliation: https://fmlab.hse.ru/
"""

from .core.universe import *
from .core.benchmark import *
from .core.factor import *
from .core.portfolio import *
from .core import factor_model as fm

from .analytics import dashboards as dash
from .analytics import metrics
from .analytics import tests
from .analytics import regressions

from . import utils
