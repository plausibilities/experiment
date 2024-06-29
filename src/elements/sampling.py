import typing

import config

class Sampling(typing.NamedTuple):
    """

    Attributes
    ----------

    draws : int
        The number of samples, i.e., data instances, to draw.

    tune : int
        The number of tuning instances.

    chains : int
        If the _chain method_ of a GPU (graphics processing unit) computation setting is
        * Parallel, $\le$ # of graphics processing units
        * Vectorized, try $\ge 4$.  Samples will be drawn in parallel within GPU.

    cores : int
        The "... number of chains to run in parallel".  Logically
            minimum(# of central processing unit cores, 4)

    random_seed : int
        Seed

    target_accept : float
        https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sample.html

    chain_method : str
        The chain method is either _parallel_ or _vectorized_
        * [blackjax](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sampling.jax.sample_blackjax_nuts.html)
        * [numpyro](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sampling.jax.sample_numpyro_nuts.html)

    """

    draws: int = 2000
    tune: int = 1000
    chains: int = 4
    cores: int = 4
    random_seed: int = config.Config().random_seed
    target_accept: float = 0.9
    chain_method: str = 'vectorized'
