import torch
import torch.nn as nn
import torch.nn.functional as F

class NNModelConv(nn.Module):

    def __init__(self, ni, no):

        super(NNModelConv, self).__init__()

        self.c1 = nn.Conv1d(1, 30, stride = no, kernel_size=3*no)
        self.p1 = nn.AvgPool1d(2)
        self.c2 = nn.Conv1d(30, 30, stride = 1, kernel_size=3)
        self.p2 = nn.AvgPool1d(2)
        self.fc1 = nn.Linear(30, no)

    def forward(self, x):
        rows, _, _ = x.size()
        x = F.relu(self.c1(x))
        x = F.relu(self.p1(x))
        x = F.relu(self.c2(x))
        x = F.relu(self.p2(x))
        x = x.view((rows,-1))
        x = self.fc1(x)
        return x
