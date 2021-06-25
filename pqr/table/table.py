import numpy as np


class Table:
    values: np.ndarray
    index: np.ndarray
    columns: np.ndarray
    name: str

    def __init__(self,
                 values: np.ndarray,
                 index: np.ndarray,
                 columns: np.ndarray,
                 name: str = ''):
        self.values = values
        self.index = index
        self.columns = columns
        self.name = name

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.name})'

    @property
    def values(self) -> np.ndarray:
        return self._values

    @values.setter
    def values(self, value: np.ndarray) -> None:
        if isinstance(value, np.ndarray):
            self._values = np.array(value)
        else:
            raise ValueError('...')

    @property
    def index(self) -> np.ndarray:
        return self._index

    @index.setter
    def index(self, value: np.ndarray) -> None:
        if isinstance(value, np.ndarray):
            self._index = np.array(value)
        else:
            raise ValueError()

    @property
    def columns(self) -> np.ndarray:
        return self._columns

    @columns.setter
    def columns(self, value: np.ndarray) -> None:
        if isinstance(value, np.ndarray):
            self._columns = np.array(value)
        else:
            raise ValueError()

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        if isinstance(value, str):
            self._name = value
        else:
            raise ValueError()

    def _is_comparable(self, other: 'Table') -> bool:
        if isinstance(other, Table):
            return np.all(self.index == other.index) \
                   and np.all(self.columns == other.columns)
        return False
