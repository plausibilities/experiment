import pymc3 as pm
import arviz as az


class Inference:

    def __init__(self):
        """

        """

    def exc(self, data):

        with pm.Model() as model:

            # Priors
            # A prior distribution for the
            #   residuals/noise/errors, sigma
            #   intercept, c
            #   gradient, m
            sigma = pm.HalfCauchy(name='sigma', beta=10)
            intercept = pm.Normal(name='intercept', mu=0, sigma=20)
            gradient = pm.Normal(name='gradient', mu=0, sigma=20)

            # Hypothesis
            regression = pm.Deterministic('regression', intercept + gradient * data.independent)

            # Define likelihood
            likelihood = pm.Normal(name='y', mu=regression, sigma=sigma, observed=data.dependent)

            # Inference!
            # draw 3000 posterior samples using NUTS sampling
            trace = pm.sample(draws=3000, cores=2, tune=2000)
            maximal = pm.find_MAP()

            # The trace generated from MCMC sampling
            trace = az.from_pymc3(trace=trace)
