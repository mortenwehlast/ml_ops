import torch
import pytest
from src.models.model import Classifier


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ("torch.randn((10, 1, 28, 28))", torch.Size([10, 10])),
        ("torch.randn((1, 1, 28, 28))", torch.Size([1, 10])),
        ("torch.randn((100, 1, 28, 28))", torch.Size([100, 10])),
    ]
)
def test_model_output_shape(test_input, expected):
    model = Classifier()
    assert model(eval(test_input)).shape == expected


def test_raises_on_input_shape():
    with pytest.raises(ValueError):
        model = Classifier()
        x = torch.rand((1, 1))
        model.forward(x)

