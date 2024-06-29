"""
The main module for running other classes
"""
import logging
import os
import sys

import arviz
import jax
import pymc


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
    logger.info(data.dependent.shape)
    logger.info(data.independent.shape)
    
    # The suggested model
    model: pymc.Model = src.model.algorithm.Algorithm().exc(data=data)

    # Inference
    inference = src.model.initial.inference.Inference(model=model)
    estimates:arviz.InferenceData = inference.exc(sampler='numpyro', method='vectorized')
    logger.info(estimates.__dict__)

    sampling = smp.Sampling(chains=8)
    interface = src.model.interface.Interface(sampling=sampling)
    estimates: arviz.InferenceData = interface.exc(model=model, nuts_sampler='numpyro', method='vectorized')
    logger.info(estimates.__dict__)


    # Delete __pycache__ directories
    src.functions.cache.Cache().exc()

if __name__ == '__main__':

    # Paths
    root = os.getcwd()
    sys.path.append(root)
    sys.path.append(os.path.join(root, 'src'))

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['XLA_FLAGS'] = (
        '--xla_gpu_enable_triton_softmax_fusion=True '
        '--xla_gpu_triton_gemm_any=True '
    )

    # Logging
    logging.basicConfig(level=logging.INFO,
                        format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(__name__)

    # Classes
    import src.data.points
    import src.elements.points as pi
    import src.elements.sampling as smp
    import src.functions.cache
    import src.model.algorithm
    import src.model.initial.inference
    import src.model.interface

    main()
