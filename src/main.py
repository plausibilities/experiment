"""
The main module for running other classes
"""
import logging
import os
import sys

import pymc
import jax


def main():
    """
    Entry point
    :return: None
    """

    # Notes
    logger.info('JAX')
    logger.info(jax.devices(backend='gpu'))
    logger.info(f"The number of GPU devices: {jax.device_count(backend='gpu')}")
    
    # Sample data
    data: pi.Points  = src.data.points.Points().exc()
    
    # The suggested model
    model: pymc.Model = src.model.algorithm.Algorithm().exc(data=data)

    # The inference instance
    inference = src.model.inference.Inference(model=model)

    # Estimating the model's parameters
    objects = inference.exc(sampler='blackjax', method='vectorized')
    logger.info(objects)

    # Delete __pycache__ directories
    src.functions.cache.Cache().exc()

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
    import src.functions.cache
    import src.model.algorithm
    import src.model.inference
    import src.elements.points as pi

    main()
