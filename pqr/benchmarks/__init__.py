"""
This module contains different benchmarks. Benchmark is the theoretical
portfolio (usually stock index), which every trader dream to beat. So, it is
used to compare performance of a factor strategies and calculate different
statistical metrics. For variable usage cases created different benchmarks:
    * Benchmark - already existing benchmark, you want to add.

    * CustomBenchmark - if it is not available to provide benchmark for
research, but you need it (just buy all stock_universe with weights (by
default equal weights are used)).
"""

from .benchmark import Benchmark
from .custombenchmark import CustomBenchmark
