import torch.nn as nn

from ..builder import parse_module


class PatchRPN(nn.Module):
    def __init__(self, config):
        super(PatchRPN, self).__init__()

        self.prpn = parse_module(config)

    def forward(self, x):
        x = self.prpn(x)

        return x
