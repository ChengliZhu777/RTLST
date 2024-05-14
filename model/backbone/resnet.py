import torch.nn as nn

from ..builder import parse_module


class ResNet(nn.Module):
    def __init__(self, config):
        super(ResNet, self).__init__()

        self.model = parse_module(config)

    def forward(self, x):
        return self.model(x)
        
