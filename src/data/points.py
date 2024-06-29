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
        self.__parameters = pr.Parameters()

    def __model(self) -> (np.ndarray, np.ndarray):
        """

        :return:
            abscissae: An original set of x values.
            ordinates: The corresponding y values abscissae.
        """

        abscissae = np.linspace(start=0, stop=2, num=self.__parameters.n_instances)
        abscissae = np.expand_dims(abscissae, axis=1)

        ordinates = (self.__parameters.gradient * abscissae) + self.__parameters.intercept

        return abscissae, ordinates

    def __measures(self, abscissae: np.ndarray, ordinates: np.ndarray) -> (np.ndarray, np.ndarray):
        """

        :param abscissae: An original set of x values.
        :param ordinates: The corresponding y values abscissae.
        :return:
            independent:  An excerpt of abscissae.
            dependent: The corresponding **noisy y values** of independent.
        """

        # Noise
        noise = np.random.normal(
            loc=self.__parameters.noise_location, scale=self.__parameters.noise_scale, size=self.__parameters.n_excerpt)
        noise = np.expand_dims(noise, axis=1)

        # The Measures
        independent = abscissae[:self.__parameters.n_excerpt]
        dependent = ordinates[:self.__parameters.n_excerpt] + noise # [:self.__parameters.n_excerpt]

        return independent, dependent

    def exc(self) -> pi.Points:
        """

        :return:
            abscissae: An original set of x values.
            ordinates: The corresponding y values abscissae.
            independent:  An excerpt of abscissae.
            dependent: The corresponding **noisy y values** of independent.
        """

        abscissae, ordinates = self.__model()
        independent, dependent = self.__measures(abscissae=abscissae, ordinates=ordinates)

        return pi.Points(abscissae=abscissae, ordinates=ordinates,
                         independent=independent, dependent=dependent)
