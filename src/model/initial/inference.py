"""
inference.py
"""
import logging

import arviz
import pymc

import src.model.initial.blackjax
import src.model.initial.numpyro
import src.model.initial.simple


class Inference:

    def __init__(self, model: pymc.Model):
        """
        
        :param model:
        """

        self.__model = model

        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def exc(self, sampler: str, method: str = 'parallel') -> arviz.InferenceData:
        """
        The chain method refers to the sample drawing method.  In the case of NumPyro the options are parallel,
        sequential, and vectorized.  Whereas, it is parallel or vectorized in the case of BlackJAX

        * blackjax: https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sampling.jax.sample_blackjax_nuts.html
        * numpyro: https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sampling.jax.sample_numpyro_nuts.html

        :param sampler:
        :param method:
        :return:
        """

        match sampler:
            case 'simple':
                return src.model.initial.simple.Simple().exc(model=self.__model)
            case 'numpyro':
                return src.model.initial.numpyro.NumPyro().exc(model=self.__model, method=method)
            case 'blackjax':
                return src.model.initial.blackjax.BlackJAX().exc(model=self.__model, method=method)
            case _:
                raise f'{sampler} is not a known sampler.'
