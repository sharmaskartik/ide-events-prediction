import torch
import torch.nn as nn
import torch.nn.functional as F

class NNClassifier(nn.Module):

    def __init__(self, ni, no):

        super(NNClassifier, self).__init__()

        self.fc1 = nn.Linear(ni, 30)
        self.fc2 = nn.Linear(30, 10)
        self.fc3 = nn.Linear(10, no)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
