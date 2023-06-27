"""
The main module for running other classes
"""
import logging
import os
import sys

import pymc


def main():
    """
    Entry point
    :return: None
    """

    logger.info('experiment')

    data: config.Config().DataCollection = src.data.points.Points().exc()
    model: pymc.Model = src.model.inference.Inference().exc(data=data)


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
    import src.model.inference

    main()
