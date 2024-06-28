"""Module points.py"""
import numpy as np

import src.elements.parameters as pr
import src.elements.points as pi


class Points:
    """
    Builds a simple set of data points
    """

    def __init__(self):
        """
        Constructor
        """

        # Parameters for data generation
        self.parameters = pr.Parameters

    def __model(self) -> (np.ndarray, np.ndarray):
        """

        :return:
        """

        abscissae = np.linspace(start=0, stop=2, num=self.parameters.n_instances)
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
            loc=self.parameters.noise_location, scale=self.parameters.noise_scale, size=self.parameters.n_excerpt)
        noise = np.expand_dims(noise, axis=1)

        # The Measures
        independent = abscissae[:self.parameters.n_excerpt]
        dependent = ordinates[:self.parameters.n_excerpt] + noise[:self.parameters.n_excerpt]

        return independent, dependent

    def exc(self) -> pi.Points:
        """

        :return:
        """

        abscissae, ordinates = self.__model()
        independent, dependent = self.__measures(abscissae=abscissae, ordinates=ordinates)

        return pi.Points(abscissae=abscissae, ordinates=ordinates,
                         independent=independent, dependent=dependent)
