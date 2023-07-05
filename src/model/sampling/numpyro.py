import os

import arviz
import pymc
import pymc.sampling_jax

import config


class NumPyro:

    def __init__(self):
        """

        """

        # Use GPU (Graphics Processing Unit) 1; the NVIDIA unit.
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        # Configurations
        configurations = config.Config()
        self.random_seed = configurations.random_seed

    def exc(self, model: pymc.model.Model, chain_method: str) -> arviz.InferenceData:
        """

        :param model:
        :param chain_method:
        :return:
        """

        with model:
            trace = pymc.sampling_jax.sample_numpyro_nuts(
                draws=2000, tune=1000, chains=4, target_accept=0.9,
                random_seed=self.random_seed, chain_method=chain_method)

        return trace
