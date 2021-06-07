import os
from pathlib import Path

import numpy as np
import torch

from src.data.make_dataset import load_mnist
from src.models.model import Classifier


def load_model(path=None):
    ROOT = str(Path(__file__).parent.parent.parent)

    if path:
        try:
            model = torch.load(Path)
        except Exception as e:
            print("No valid model found at path. Python error:")
            print(e)
            exit()
    else:  # Else load newest model
        # Check if trained model exists
        model_paths = [
            d
            for d in os.listdir(os.path.join(ROOT, "models", "."))
            if d.endswith(".pth")
        ]
        if model_paths:  # Load newest model (based on initial time stamp)
            model_path = sorted(model_paths)[0]
            model = Classifier()
            model.load_state_dict(
                torch.load(os.path.join(ROOT, "models", model_path))
            )
        else:
            print("No models trained at this point in time. Exiting")
            exit()

    return model


def load_test_set(path=None):
    if path:
        try:
            if path.endswith(".pkl"):
                test_set = torch.load(path)
            elif path.endswith(".npz"):
                test_set = torch.tensor(np.load(path))
            else:
                print("Files must be .pkl or .npz format.")
                raise IOError

        except Exception as e:
            print("Not a valid path for test set. Python error:")
            print(e)
            exit()
    else:
        _, test_set = load_mnist()

    return test_set
