class RTLSTD(nn.Module):
    def __init__(self, patch_size, backbone, rpn):
        super(RTLSTD, self).__init__()
      self.encoder = build_model(backbone)

    def forward(self, x):
        rpn = self.encoder(patch_images)
        
