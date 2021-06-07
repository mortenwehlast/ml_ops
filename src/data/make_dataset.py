# -*- coding: utf-8 -*-
import logging
from pathlib import Path

from torchvision import datasets, transforms


def load_mnist():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    ROOT = str(Path(__file__).parent.parent.parent)

    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    # Download and load the training data
    trainset = datasets.FashionMNIST(
        ROOT + '/data/F_MNIST_train',
        download=True,
        train=True,
        transform=transform
    )

    # Download and load the test data
    testset = datasets.FashionMNIST(
        ROOT + '/data/F_MNIST_test',
        download=True, train=False,
        transform=transform
    )

    return trainset, testset


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    load_mnist()
