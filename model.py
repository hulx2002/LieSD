import torch
from utils import *

class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity='tanh', classify=False):
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, output_dim)

        self.nonlinearity = choose_nonlinearity(nonlinearity)

        self.classify = classify

    def forward(self, x):
        h = self.nonlinearity(self.linear1(x))
        h = self.nonlinearity(self.linear2(h))
        h = self.linear3(h)
        if self.classify:
            return torch.sigmoid(h)
        else:
            return h

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = torch.nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)
        self.pool = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.nonlinearity = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.nonlinearity(self.conv1(x)))
        x = self.pool(self.nonlinearity(self.conv2(x)))
        x = x.view(x.shape[0], 16 * 5 * 5)
        x = self.nonlinearity(self.fc1(x))
        x = self.nonlinearity(self.fc2(x))
        x = self.fc3(x)
        return x
        