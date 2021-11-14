from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from pqr.core import Portfolio
from .graph import Graph
from .table import Table


@dataclass
class Dashboard:
    items: Sequence[Graph | Table]

    def __call__(self, portfolios: Sequence[Portfolio]) -> None:
        for item in self.items:
            item(portfolios)
