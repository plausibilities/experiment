import pymc


class Inference:

    def __init__(self):
        """

        """

    def exc(self, data) -> pymc.Model:

        with pymc.Model() as model:

            # Priors
            # A prior distribution for the
            #   residuals/noise/errors, sigma
            #   intercept, c
            #   gradient, m
            sigma = pymc.HalfCauchy(name='sigma', beta=10)
            intercept = pymc.Normal(name='intercept', mu=0, sigma=20)
            gradient = pymc.Normal(name='gradient', mu=0, sigma=20)

            # Hypothesis
            regression = pymc.Deterministic('regression', intercept + gradient * data.independent)

            # Define likelihood
            likelihood = pymc.Normal(name='y', mu=regression, sigma=sigma, observed=data.dependent)

        return model
