from typing import Union

import numpy as np


class HasNameMixin:
    _name: Union[str, None]

    def __init__(self, name):
        self.name = name

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(' \
               f'{self.name if self.name is not None else ""})'

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        if isinstance(value, str) or value is None:
            self._name = value
        else:
            raise ValueError('name must be str')


class HasIndexMixin:
    _index: np.ndarray

    def __init__(self):
        self.index = np.array([])

    @property
    def index(self) -> np.ndarray:
        return self._index

    @index.setter
    def index(self, value: np.ndarray) -> None:
        if isinstance(value, np.ndarray):
            self._index = np.array(value)
        else:
            raise ValueError('index must be np.ndarray')
