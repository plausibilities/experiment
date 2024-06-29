"""
simple.py
"""
import arviz
import pymc

import config


class Simple:

    def __init__(self):
        """
        Constructor
        """

        # Configurations
        configurations = config.Config()
        self.random_seed = configurations.random_seed

    def exc(self, model: pymc.model.Model) -> arviz.InferenceData:
        """

        :param model:
        :return:
        """

        with model:

            # Drawing samples using NUTS sampling
            trace = pymc.sample(draws=2000, tune=1000, chains=4, cores=4, target_accept=0.9,
                                random_seed=self.random_seed, nuts_sampler='pymc',
                                nuts_sampler_kwargs=None)

        return trace
