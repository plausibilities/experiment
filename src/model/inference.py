import jax
import arviz

import pymc

import src.elements.sampling as smp


class Interference:

    def __init__(self, sampling: smp.Sampling):

        self.__sampling = sampling

    def __chains(self, method: str) -> int:
        """
        Ensures the chains value is in line with processing units
        numbers, and computation logic.

        :param method:
        :return:
        """

        if method == 'parallel':
            return jax.device_count(backend='gpu')
        else:
            return self.__sampling.chains

    @staticmethod
    def __nuts_sampler_kwargs(nuts_sampler: str, method: str):
        """
        Sets the NUTS sampling dictionary for BlackJax or Numpyro

        :param nuts_sampler:
        :param method:
        :return:
        """

        if nuts_sampler in ['blackjax', 'numpyro']:
            return {'chain_method': method,
                    'postprocessing_backend': 'gpu'}
        else:
            return None

    # noinspection PyTypeChecker
    def exc(self, model: pymc.model.Model, nuts_sampler: str, method: str)  -> arviz.InferenceData:
        """

        :param model:
        :param nuts_sampler: pymc, numpyro, blackjax
        :param method: parallel, vectorized
        :return:
        """

        # The BlackJax progress bar fails
        if nuts_sampler == 'blackjax':
            progressbar = False
        else:
            progressbar = True

        # Proceed
        with model:

            trace = pymc.sample(
                draws=self.__sampling.draws,
                tune=self.__sampling.tune,
                chains=self.__chains(method=method),
                cores=self.__sampling.cores,
                target_accept=self.__sampling.target_accept,
                random_seed=self.__sampling.random_seed,
                nuts_sampler=nuts_sampler,
                nuts_sampler_kwargs=self.__nuts_sampler_kwargs(nuts_sampler=nuts_sampler, method=method),
                progressbar=progressbar)

        return trace
