from abc import abstractmethod

import pandas as pd


class IWeighting:
    """
    Class-interface for weighting factors.
    """

    @abstractmethod
    def weigh(self, data: pd.DataFrame) -> pd.DataFrame:
        ...
