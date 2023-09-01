import logging
import os
import sys

def main():

    logger.info('Sampling Options')



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
    import src.model.algorithm
    import src.model.inference

    # The inference options
    samplers = ['numpyro', 'numpyro', 'blackjax']
    methods = ['parallel', 'vectorized', 'vectorized']

    main()