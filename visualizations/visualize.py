import argparse
import sys

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.manifold import TSNE

from src.utils.utils import load_model, load_test_set

palette = sns.color_palette("bright", 10)


def get_activation(name, activation):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


print("Embedding for visualization")
parser = argparse.ArgumentParser(description='Embedding arguments')
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

# set hook:
activation = {}
model.fc3.register_forward_hook(get_activation("fc3", activation))

# Embed
all_labels = []
all_outputs = []

with torch.no_grad():
    model.eval()
    res = torch.zeros(0)
    for images, labels in testloader:
        model(images)
        outputs = activation["fc3"]

        all_labels += labels.tolist()
        all_outputs += outputs.tolist()

mnist_embedded = TSNE(
    n_components=2,
).fit_transform(all_outputs)

plt.figure(figsize=(8, 5))
sns_plot = sns.scatterplot(
    mnist_embedded[:, 0],
    mnist_embedded[:, 1],
    hue=all_labels,
    palette=palette,
    legend="full",
)
plt.title("FMNIST T-SNE plot")
plt.savefig("reports/figures/tsne_plot.png")
