import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision
from model import Classifier
from torch import nn, optim

from src.data.make_dataset import load_mnist

print("Training day and night")
parser = argparse.ArgumentParser(description='Training arguments')
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--dropout', default=0.2, type=float)
parser.add_argument('--save_plots', default=True, type=bool)

# add any additional argument that you want
args = parser.parse_args(sys.argv[2:])
print(args)

# Define hyperparams
epochs = args.epochs
dropout = args.dropout
lr = args.lr
batch_size = args.batch_size

# Set root path
ROOT = str(Path(__file__).parent.parent.parent)

# Get training data
train_set, _ = load_mnist()
trainloader = torch.utils.data.DataLoader(
    train_set,
    batch_size=args.batch_size,
    shuffle=True
)

# Get num inputs and classes and dropout
temp_elem, _ = train_set[0]
n_inputs = torch.numel(temp_elem)
n_classes = len(train_set.classes)

# Specify model
model = Classifier(
    n_inputs=n_inputs,
    n_classes=n_classes,
    dropout=dropout
)


# wandb
import wandb
wandb.init(config=args)

# Magic
wandb.watch(model, log_freq=100)

# Set optimizer, loss and learning rate
optimizer = optim.SGD(model.parameters(), lr=lr)
criterion = nn.NLLLoss()
model.train()

# Init for tracking of batch losses with histogram

for e in range(epochs):
    epoch_loss = 0
    batch_losses = torch.zeros(len(trainloader))
    for idx, (images, labels) in enumerate(trainloader):
        optimizer.zero_grad()

        # Compute scores and loss
        scores = model(images)
        loss = criterion(scores, labels)
        epoch_loss += loss

        # Backpropagate
        loss.backward()

        # Optimize
        optimizer.step()

        batch_losses[idx] = loss.item()

    wandb.log({"loss": epoch_loss.item()})
    print(f"Training loss for epoch {e+1}: {epoch_loss}")

wandb.log({"examples" : [wandb.Image(i) for i in images]})
wandb.finish()

print("Training complete.")