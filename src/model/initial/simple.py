"""
simple.py
"""
import os
import arviz
import pymc
import multiprocessing

import config


class Simple:

    def __init__(self):
        """
        Constructor
        """

        # Use a CPU (Central Processing Unit)
        # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        # Configurations
        configurations = config.Config()
        self.random_seed = configurations.random_seed

    def exc(self, model: pymc.model.Model) -> arviz.InferenceData:
        """

        :param model:
        :return:
        """

        # This step addresses an upcoming Python development that addresses incompatibilities
        # between os.fork() and multithreaded programs
        # https://docs.python.org/3/library/os.html#os.fork
        multiprocessing.set_start_method(method='spawn', force=True)

        # Proceed
        with model:

            # Drawing samples using NUTS sampling
            trace = pymc.sample(draws=2000, tune=1000, chains=4, cores=4, target_accept=0.9,
                                random_seed=self.random_seed, nuts_sampler='pymc',
                                nuts_sampler_kwargs=None)

        return trace
