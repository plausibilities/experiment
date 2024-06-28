"""
This is the data type Parameters
"""
import typing


class Parameters(typing.NamedTuple):
    """
    The data type class ⇾ Parameters
    
    T=600, N=500, intercept=1.5, gradient=2.5, noise_loc=0.0, noise_scale=0.5
    n_instances, n_excerpt, intercept, gradient, noise_location, noise_scale

    Attributes
    ----------
      n_instances :
        The number of instances.

      n_excerpt :
        An excerpt's length.

      intercept :
        The intercept c of y = mx + c.

      gradient :
        The gradient m of y = mx + c.

      noise_location :
        The mean of a Gaussian distribution

      noise_scale :
        The standard deviation of a Gaussian distribution.

    """

    n_instances: int = 600
    n_excerpt: int = 500
    intercept: float = 1.5
    gradient: float = 2.5
    noise_location: float = 0.0
    noise_scale: float = 0.5
