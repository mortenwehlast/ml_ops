import torch
from src.models.model import Classifier

def test_model_output_shape():
    model = Classifier()
    image = torch.randn((1, 28, 28))
    assert model(image).shape == torch.Size([1, 10])
