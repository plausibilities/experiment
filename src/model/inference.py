"""Module inference.py"""

import arviz
import jax
import pymc

import src.elements.sampling as smp


class Inference:
    """
    Inference
    """

    def __init__(self, sampling: smp.Sampling):
        """

        :param sampling:
        """

        self.__sampling = sampling

        self.__nuts_samplers = ['blackjax', 'numpyro', 'pymc']
        self.__graphics_nuts_samplers = ['blackjax', 'numpyro']
        self.__graphics_chain_methods = ['parallel', 'vectorized']

    def __chains(self, chain_method: str) -> int:
        """
        Ensures the chains value is in line with processing units
        numbers, and computation logic.

        :param chain_method:
        :return:
        """

        if chain_method == 'parallel':
            return jax.device_count(backend='gpu')

        return self.__sampling.chains

    def __nuts_sampler_kwargs(self, nuts_sampler: str, chain_method: str):
        """
        Sets the sampling dictionary for BlackJax or Numpyro

        :param nuts_sampler:
        :param chain_method:
        :return:
            A dict of arguments for Numpyro or BlackJax
        """

        if nuts_sampler in self.__graphics_nuts_samplers:
            return {'chain_method': chain_method,
                    'postprocessing_backend': 'gpu'}

        return None

    def __inspect(self, nuts_sampler: str, chain_method: str) -> bool:
        """

        :param nuts_sampler: A NUTS (No U Turn Sampler) type

        :param chain_method: A chain method type
        :return:
            None
        """

        if chain_method not in self.__graphics_chain_methods:
            raise Exception('Unknown graphics chain method; parallel or vectorized only.')

        if nuts_sampler not in self.__nuts_samplers:
            raise Exception('Unknown graphics chain method; parallel or vectorized only.')

        return True

    # noinspection PyTypeChecker
    def exc(self, model: pymc.model.Model, nuts_sampler: str, chain_method: str)  -> arviz.InferenceData:
        """

        :param model:
        :param nuts_sampler: Either pymc, numpyro, or blackjax
        :param chain_method: Either parallel or vectorized.  Applies to Numpyro & BlackJax only.
        :return:
            arviz.InferenceData
        """

        # Ascertain NUTS, and chain method, settings.
        self.__inspect(nuts_sampler=nuts_sampler, chain_method=chain_method)

        # At present, the BlackJax progress bar fails
        if nuts_sampler == 'blackjax':
            progressbar = False
        else:
            progressbar = True

        # Proceed
        with model:
            trace = pymc.sample(
                draws=self.__sampling.draws,
                tune=self.__sampling.tune,
                chains=self.__chains(chain_method=chain_method),
                cores=self.__sampling.cores,
                target_accept=self.__sampling.target_accept,
                random_seed=self.__sampling.random_seed,
                nuts_sampler=nuts_sampler,
                nuts_sampler_kwargs=self.__nuts_sampler_kwargs(nuts_sampler=nuts_sampler, chain_method=chain_method),
                progressbar=progressbar)

        return trace
