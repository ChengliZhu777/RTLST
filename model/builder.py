import models


def build_model(cfg):
    param = dict()
    for key in cfg:
        if key in ['type', 'is_recognize']:
            continue
        param[key] = cfg[key]

    if cfg['type'] in ['RTLSTD']:
        return models.__dict__[cfg['type']](**param)
    elif cfg in ['ResNet']:
        return models.backbone.__dict__[cfg['type']](**param)
