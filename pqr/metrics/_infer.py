from __future__ import annotations

__all__ = [
    "infer",
]

from functools import wraps
from typing import Callable

import pandas as pd

from pqr.core import Portfolio, Benchmark
from pqr.utils import (
    estimate_annualizer,
    estimate_window,
)


def infer(
        returns: bool = False,
        benchmark: bool = False,
        holdings: bool = False,
        annualizer: bool = False,
        window: bool = False,
) -> Callable:
    def infered_metric(metric: Callable):
        @wraps(metric)
        def inferer(*args, **kwargs):
            if returns:
                args, kwargs = _infer_returns(*args, **kwargs)
            if benchmark:
                args, kwargs = _infer_benchmark(*args, **kwargs)
            if holdings:
                args, kwargs = _infer_holdings(*args, **kwargs)

            if annualizer:
                kwargs = _infer_annualizer(*args, **kwargs)
            if window:
                kwargs = _infer_window(*args, **kwargs)

            result = metric(*args, **kwargs)
            if isinstance(result, pd.Series):
                result.name = metric.__name__

            return result

        return inferer

    return infered_metric


def _infer_returns(*args, **kwargs):
    args = list(args)
    if args:
        if isinstance(args[0], (Portfolio, Benchmark)):
            args[0] = args[0].returns
    else:
        if isinstance(kwargs["returns"], (Portfolio, Benchmark)):
            kwargs["returns"] = kwargs["returns"].returns

    return tuple(args), kwargs


def _infer_benchmark(*args, **kwargs):
    args = list(args)
    if len(args) > 1:
        if isinstance(args[1], (Portfolio, Benchmark)):
            args[1] = args[1].returns
    else:
        if isinstance(kwargs["benchmark"], (Portfolio, Benchmark)):
            kwargs["benchmark"] = kwargs["benchmark"].returns

    return tuple(args), kwargs


def _infer_holdings(*args, **kwargs):
    args = list(args)
    if args:
        if isinstance(args[0], Portfolio):
            args[0] = args[0].holdings
        elif isinstance(args[0], Benchmark):
            raise ValueError("cannot get holdings from benchmark")
    else:
        if isinstance(kwargs["holdings"], Portfolio):
            kwargs["holdings"] = kwargs["holdings"].holdings
        elif isinstance(kwargs["holdings"], Benchmark):
            raise ValueError("cannot get holdings from benchmark")

    return tuple(args), kwargs


def _infer_annualizer(*args, **kwargs):
    if "annualizer" not in kwargs:
        if args:
            kwargs["annualizer"] = estimate_annualizer(args[0])
        else:
            try:
                kwargs["annualizer"] = estimate_annualizer(kwargs["returns"])
            except KeyError:
                kwargs["annualizer"] = estimate_annualizer(kwargs["holdings"])

    return kwargs


def _infer_window(*args, **kwargs):
    if "window" not in kwargs:
        if args:
            kwargs["window"] = estimate_window(args[0])
        else:
            try:
                kwargs["window"] = estimate_window(kwargs["returns"])
            except KeyError:
                kwargs["window"] = estimate_window(kwargs["holdings"])

    return kwargs
