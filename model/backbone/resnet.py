import torch.nn as nn


class ResNet(nn.Module):
    def __init__(self, config):
        super(ResNet, self).__init__()

        self.model, self.save_layers = parse_model(config, module_name='ResNet-6')
      
