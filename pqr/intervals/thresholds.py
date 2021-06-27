from .interval import Interval


class Thresholds(Interval):
    """
    Class for intervals of thresholds.

    Parameters
    ----------
    lower : int, float, default=-np.inf
        Lower threshold.
    upper : int, float, default=np.inf
        Upper threshold.

    Raises
    ------
    ValueError
        Lower threshold more than upper.
    TypeError
        A threshold isn't int or float.
    """
