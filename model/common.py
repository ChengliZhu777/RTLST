import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride,
                 padding=None, groups=1, bias=True, bn=False, act=None):
        super(Conv, self).__init__()

        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride,
                              padding=auto_padding(kernel_size, padding), groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(output_channels) if bn else None
        self.act = act

    def forward(self, x):
        out = self.conv(x)

        if self.bn is not None:
            out = self.bn(out)
        if self.act is not None:
            out = self.act(out)
        return out
      
