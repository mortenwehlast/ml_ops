import argparse
import sys

import torch

from src.utils.utils import load_model, load_test_set

print("Evaluating until hitting the ceiling")
parser = argparse.ArgumentParser(description='Prediction arguments')
parser.add_argument('--load_model_from', default="")
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--test_set_path', default="", type=str)
args = parser.parse_args(sys.argv[2:])
print(args)


# Load model
model = load_model(args.load_model_from)

# Load test set
test_set = load_test_set(args.test_set_path)

testloader = torch.utils.data.DataLoader(
    test_set,
    batch_size=args.batch_size,
    shuffle=True
)

# Set model to eval mode
model.eval()

with torch.no_grad():
    n_obs = len(testloader)
    hits = 0
    for images, labels in testloader:
        # Compute probs and extract predictions
        probs = torch.exp(model(images))
        top_p, top_class = probs.topk(1, dim=1)

        # Compare
        batch_hits = top_class == labels.view(*top_class.shape)
        hits += torch.sum(batch_hits)

    print("Test Accuracy: {:.2f}%".format(hits/n_obs))
