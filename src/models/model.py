from torch import nn


class Classifier(nn.Module):
    def __init__(self, n_inputs=784, n_classes=10, dropout=0.2):
        super().__init__()

        self.n_inputs = n_inputs
        self.n_classes = n_classes

        self.fc1 = nn.Linear(n_inputs, 364)
        self.fc2 = nn.Linear(364, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, n_classes)
        self.out = nn.LogSoftmax(dim=1)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError('Expected input to a 4D tensor')
        if x.shape[1] != 1 or x.shape[2] != 28 or x.shape[3] != 28:
            raise ValueError('Expected each sample to have shape [1, 28, 28]')

        # Ensure x is flattened
        x = x.view(x.shape[0], -1)

        x = self.act(self.fc1(x))
        x = self.dropout(self.act(self.fc2(x)))
        x = self.act(self.fc3(x))
        x = self.dropout(self.act(self.fc4(x)))
        x = self.out(x)

        return x
