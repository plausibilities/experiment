"""
The main module for running other classes
"""
import logging
import os
import sys
import time

import jax
import pymc


def main():
    """
    Entry point
    :return: None
    """

    # Notes
    logger.info('experiment')
    logger.info(jax.default_backend())
    logger.info(jax.devices())

    # Sample data
    data: config.Config().DataCollection = src.data.points.Points().exc()

    # The suggested model
    model: pymc.Model = src.model.algorithm.Algorithm().exc(data=data)

    # The inference instance
    inference = src.model.inference.Inference(model=model)

    # Estimating the model's parameters via different sampling methods
    for sampler, method in zip(samplers, methods):
        logger.info('%s ...', sampler)
        starts = time.time()
        tablet = inference.exc(sampler=sampler, method=method)
        logger.info(tablet)
        logger.info(f'{sampler}: {time.time() - starts}')


if __name__ == '__main__':

    # Paths
    root = os.getcwd()
    sys.path.append(root)
    sys.path.append(os.path.join(root, 'src'))

    # Logging
    logging.basicConfig(level=logging.INFO,
                        format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(__name__)

    # Classes
    import config
    import src.data.points
    import src.model.algorithm
    import src.model.inference

    # The inference options
    samplers = ['numpyro', 'blackjax', 'numpyro', 'blackjax']
    methods = ['vectorized', 'vectorized', 'parallel', 'parallel']

    main()
