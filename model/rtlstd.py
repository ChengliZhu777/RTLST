class RTLSTD(nn.Module):
    def __init__(self, patch_size, backbone, rpn):
        super(RTLSTD, self).__init__()
        self.patch_size = patch_size
        self.encoder = build_model(backbone)
        self.patch_rpn = build_model(rpn)

    def forward(self, x):
        rpn = self.patch_rpn(self.encoder(patch_images))
        
