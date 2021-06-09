import torch
from src.data.make_dataset import load_mnist


def test_data():
    train_set, test_set = load_mnist()
    assert len(train_set) == 60000 and len(test_set) == 10000
    assert train_set[0][0].shape == torch.Size([1, 28, 28])
    assert test_set[0][0].shape == torch.Size([1, 28, 28])
    assert len(train_set.classes) == 10
    assert len(test_set.classes) == 10
