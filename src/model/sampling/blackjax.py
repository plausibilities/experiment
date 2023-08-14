"""
blackjax.py
"""
import os

import arviz
import pymc
import pymc.sampling_jax
import jax

import config


class BlackJAX:

    def __init__(self):
        """
        Constructor
        """

        # Use GPU (Graphics Processing Unit) 1; the NVIDIA unit.
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
            chains = 8

        with model:
            # Inference
            trace = pymc.sampling_jax.sample_blackjax_nuts(
                draws=2000, tune=1000, chains=chains, target_accept=0.9,
                random_seed=self.random_seed, chain_method=method)

        return trace
