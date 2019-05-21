import torch
import torch.nn as nn
from torch.autograd import Variable

class LockedDrop(nn.Module):
    def __init__(self,dropout=0.5):
        self.dropout=dropout
        super().__init__()

    def forward(self, x):
        if not self.training or not self.dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - self.dropout)
        mask = Variable(m, requires_grad=False) / (1 - self.dropout)
        mask = mask.expand_as(x)
        return mask * x