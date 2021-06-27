from abc import abstractmethod

import pandas as pd


class IFiltering:
    """
    Class-interface for filtering factors.
    """

    @abstractmethod
    def filter(self, data: pd.DataFrame) -> pd.DataFrame:
        ...
