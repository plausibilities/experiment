"""
This is the data type Points
"""
import typing

import numpy as np


class Points(typing.NamedTuple):
    """
    The data type class ⇾ Points

    Attributes
    ----------
      abscissae :
        An original set of x values.

      ordinates :
        The corresponding y values abscissae.

      independent :
        An excerpt of abscissae.

      dependent :
        The corresponding **noisy y values** of independent.

    """

    abscissae: np.ndarray
    ordinates: np.ndarray
    independent: np.ndarray
    dependent: np.ndarray
