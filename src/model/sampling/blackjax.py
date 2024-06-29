"""
blackjax.py
"""
import os

import arviz
import jax
import pymc

import config


class BlackJAX:

    def __init__(self):
        """
        Constructor
        """

        # Use a GPU (Graphics Processing Unit); the NVIDIA unit.
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

        # Configurations
        configurations = config.Config()
        self.random_seed = configurations.random_seed

    def exc(self, model: pymc.model.Model, method: str) -> arviz.InferenceData:
        """

        :param model:
        :param method:
        :return:
        """

        if method == 'parallel':
            chains = jax.device_count(backend='gpu')
        else:
            chains = 4

        with model:
            trace = pymc.sample(draws=2000, tune=1000, chains=chains, target_accept=0.9,
                                random_seed=self.random_seed, nuts_sampler='blackjax',
                                nuts_sampler_kwargs={'chain_method': method}, progressbar=False)

        return trace
