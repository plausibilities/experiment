"""
insights.py
"""
import logging
import os
import sys
import time

import pymc


def main():
    """
    Entry point
    :return: None
    """

    logger.info('Sampling Options')

    # Sample data
    data: config.Config().DataCollection = src.data.points.Points().exc()
    logger.info(type(data))
    
    # The suggested model
    model: pymc.Model = src.model.algorithm.Algorithm().exc(data=data)

    # The inference instance
    inference = src.model.inference.Inference(model=model)

    # Estimating the model's parameters via different sampling methods
    computations = []
    for sampler, method in zip(samplers, methods):
        
        starts = time.time()
        objects = inference.exc(sampler=sampler, method=method)
        logger.info(objects)
        computations.append({'sampler': sampler, 'method': method, 'seconds': {time.time() - starts}})

    logger.info(computations)

    # Delete __pycache__ directories
    src.functions.extraneous.Extraneous().exc()


if __name__ == '__main__':

    # Paths
    root = os.getcwd()
    sys.path.append(root)
    sys.path.append(os.path.join(root, 'src'))

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Logging
    logging.basicConfig(level=logging.INFO,
                        format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(__name__)

    # Classes
    import config
    import src.data.points
    import src.functions.extraneous
    import src.model.algorithm
    import src.model.inference

    # The inference options
    samplers = ['numpyro', 'numpyro', 'blackjax']
    methods = ['parallel', 'vectorized', 'vectorized']

    main()
