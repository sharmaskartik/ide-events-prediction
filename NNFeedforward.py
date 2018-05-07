import torch
import torch.nn as nn
import torch.nn.functional as F

class NNFeedForward(nn.Module):

    def __init__(self, ni, no):

        super(NNFeedForward, self).__init__()

        self.fc1 = nn.Linear(ni, 30)
        self.fc2 = nn.Linear(30, 30)
        self.fc3 = nn.Linear(30, no)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
