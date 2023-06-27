import collections

import numpy as np

import config


class Points:

    def __init__(self):
        """

        """

        configurations = config.Config()
        self.__DataCollection = configurations.DataCollection

        # Parameters for data generation
        Parameters = collections.namedtuple(typename='Parameters',
                                            field_names=['T', 'N', 'intercept', 'gradient', 'noise_loc', 'noise_scale'])
        self.parameters = Parameters(T=600, N=500, intercept=1.5, gradient=2.5, noise_loc=0.0, noise_scale=0.5)

    def __model(self) -> (np.ndarray, np.ndarray):
        """

        :return:
        """

        abscissae = np.linspace(start=0, stop=2, num=self.parameters.T)
        abscissae = np.expand_dims(abscissae, axis=1)

        ordinates = (self.parameters.gradient * abscissae) + self.parameters.intercept
        ordinates = np.expand_dims(ordinates, axis=1)

        return abscissae, ordinates

    def __measures(self, abscissae: np.ndarray, ordinates: np.ndarray) -> (np.ndarray, np.ndarray):
        """

        :param abscissae:
        :param ordinates:
        :return:
        """

        # Noise
        noise = np.random.normal(
            loc=self.parameters.noise_loc, scale=self.parameters.noise_scale, size=self.parameters.N)
        noise = np.expand_dims(noise, axis=1)

        # The Measures
        independent = abscissae[:self.parameters.N]
        dependent = ordinates[:self.parameters.N] + noise[:self.parameters.N]

        return independent, dependent

    def exc(self) -> config.Config().DataCollection:
        """

        :return:
        """

        abscissae, ordinates = self.__model()
        independent, dependent = self.__measures(abscissae=abscissae, ordinates=ordinates)

        return self.__DataCollection(abscissae=abscissae, ordinates=ordinates,
                                     independent=independent, dependent=dependent)
