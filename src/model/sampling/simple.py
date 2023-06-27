import arviz
import pymc
import pymc.sampling_jax

import config


class Simple:

    def __init__(self):
        """

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

            # Inference
            # draw 4000 posterior samples using NUTS sampling
            trace = pymc.sample(draws=4000, tune=2000, chains=4, target_accept=0.9,
                                random_seed=self.random_seed, nuts_sampler='pymc')

        return trace