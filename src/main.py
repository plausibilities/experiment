"""
The main module for running other classes
"""
import logging
import os
import sys
import collections
import time
import pymc


def main():
    """
    Entry point
    :return: None
    """

    logger.info('experiment')

    # Sample data
    data: config.Config().DataCollection = src.data.points.Points().exc()

    # The suggested model
    model: pymc.Model = src.model.algorithm.Algorithm().exc(data=data)

    # The inference instance
    inference = src.model.inference.Inference(model=model)

    # Estimating the model's parameters via different sampling methods
    for option in options:
        logger.info('%s ...', option.sampler)
        starts = time.time()
        inference.exc(sampler=option.sampler, chain_method=option.chain_method)
        logger.info('%s: %s', (option.sampler, time.time() - starts))


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
    Case = collections.namedtuple(typename='Case', field_names=['sampler', 'chain_method'])
    cases = [{'sampler': 'simple', 'chain_method': ''},
             {'sampler': 'numpyro', 'chain_method': 'vectorized'},
             {'sampler': 'blackjax', 'chain_method': 'vectorized'},
             {'sampler': 'numpyro', 'chain_method': 'parallel'},
             {'sampler': 'blackjax', 'chain_method': 'parallel'}]
    options = [Case(**case) for case in cases]

    main()
