import torch.nn as nn

class Squeeze(nn.Module): # for classifier
    def forward(self, x):
        return x.squeeze((-2, -1))