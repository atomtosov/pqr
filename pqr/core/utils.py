from __future__ import annotations

import functools as ft
from typing import TypeVar, Callable

import numpy as np

T = TypeVar("T")


def compose(*functions: Callable[[T], T]) -> Callable[[T], T]:
    return ft.reduce(
        lambda f, g: lambda x: g(f(x)),
        functions
    )


def normalize(raw_weights: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.nan_to_num(
            raw_weights / np.nansum(raw_weights, axis=1, keepdims=True, dtype=float),
            nan=0, neginf=0, posinf=0
        )


def top_single(arr: np.ndarray, n: np.ndarray) -> np.ndarray:
    arr = np.unique(arr[~np.isnan(arr)])
    if arr.any():
        arr = np.sort(arr)
        max_len = len(arr)
        return np.array(
            [arr[-min(_n, max_len)] for _n in n]
        )
    return np.array([np.nan] * len(n))


def bottom_single(arr: np.ndarray, n: np.ndarray) -> np.ndarray:
    arr = np.unique(arr[~np.isnan(arr)])
    if arr.any():
        arr = np.sort(arr)
        max_len = len(arr)
        return np.array(
            [arr[min(_n, max_len) - 1] for _n in n]
        )
    return np.array([np.nan] * len(n))
