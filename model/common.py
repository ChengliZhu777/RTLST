import torch.nn as nn


def auto_padding(kernel_size, padding=None):
    if padding is None:
        padding = kernel_size // 2 if isinstance(kernel_size, int) else tuple([ks // 2 for ks in kernel_size])
    return padding


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


class MaxPool(nn.Module):
    def __init__(self, kernel_size, stride, padding=0, dilation=1):
        super(MaxPool, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride,
                                     padding=padding, dilation=dilation)

    def forward(self, x):
        return self.max_pool(x)
        
