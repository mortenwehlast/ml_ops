import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from src.models.model import Classifier
from src.data.make_dataset import load_mnist


class TrainWorker():
    def __init__(self):
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.1, type=float)
        parser.add_argument('--epochs', default=10, type=int)
        parser.add_argument('--batch_size', default=64, type=int)
        parser.add_argument('--dropout', default=0.2, type=float)

        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)

        # Define hyperparams
        self.epochs = args.epochs
        self.dropout = args.dropout
        self.lr = args.lr
        self.batch_size = args.batch_size

        # Init for tracking of batch losses with histogram
        self.loss_history = []

        # Set root path
        self.ROOT = str(Path(__file__).parent.parent.parent)

    def train(self):
        # Get training data
        train_set, _ = load_mnist()
        trainloader = torch.utils.data.DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True
        )

        # Get num inputs and classes and dropout
        temp_elem, _ = train_set[0]
        n_inputs = torch.numel(temp_elem)
        n_classes = len(train_set.classes)

        # Specify model
        self.model = Classifier(
            n_inputs=n_inputs,
            n_classes=n_classes,
            dropout=self.dropout
        )

        # Set optimizer, loss and learning rate
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        criterion = nn.NLLLoss()
        self.model.train()

        for e in range(self.epochs):
            epoch_loss = 0
            batch_losses = torch.zeros(len(trainloader))
            for idx, (images, labels) in enumerate(trainloader):
                optimizer.zero_grad()

                # Compute scores and loss
                scores = self.model(images)
                loss = criterion(scores, labels)
                epoch_loss += loss

                # Backpropagate
                loss.backward()

                # Optimize
                optimizer.step()

                batch_losses[idx] = loss.item()

            self.loss_history.append(epoch_loss.item())
            print(f"Training loss for epoch {e+1}: {epoch_loss}")

        print("Training complete.")

        return batch_losses

    def save_model(self):
        # Save model
        today = datetime.now().strftime("%Y%m%d")
        filename = today + ".pth"
        path = os.path.join(self.ROOT, "models", filename)
        print("Saving model to: ", path)
        torch.save(self.model.state_dict(), path)

    def make_plots(self):
        if not self.loss_history:
            raise ValueError(
                "No loss history to be plotted. Run .train() first."
            )

        today = datetime.now().strftime("%Y%m%d")
        filename = today + ".png"
        path = os.path.join(
                self.ROOT,
                "reports",
                "figures",
                filename
            )

        plt.figure(figsize=(12, 8))
        plt.plot(list(range(self.epochs)), self.loss_history)
        plt.title(
            "Loss for FMNIST classifier with dropout "
            + f"={self.dropout} and {self.epochs} epochs."
        )
        plt.xlabel("Epoch")
        plt.ylabel("Negative Log-Likelihood")
        plt.savefig(path)
