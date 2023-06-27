import logging

import arviz
import pymc

import src.model.sampling.blackjax
import src.model.sampling.numpyro
import src.model.sampling.simple


class Inference:

    def __init__(self, model: pymc.Model):
        """

        """

        self.__model = model

        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def exc(self, sampler: str, chain_method: str = 'parallel') -> arviz.InferenceData:
        """
        The chain method refers to the sample drawing method.  In the case of NumPyro the options are parallel,
        sequential, and vectorized.  Whereas, it is parallel or vectorized in the case of BlackJAX

        * blackjax: https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sampling.jax.sample_blackjax_nuts.html
        * numpyro: https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sampling.jax.sample_numpyro_nuts.html

        :param sampler:
        :param chain_method:
        :return:
        """

        return {
            'simple': src.model.sampling.simple.Simple().exc(model=self.__model),
            'numpyro': src.model.sampling.numpyro.NumPyro().exc(model=self.__model, chain_method=chain_method),
            'blackjax': src.model.sampling.blackjax.BlackJAX().exc(model=self.__model, chain_method=chain_method)
        }.get(sampler, LookupError(f'{sampler} is not a known sampler.'))
